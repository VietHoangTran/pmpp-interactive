// Self-test: host-side runtime calls. In the emulator they are all synchronous.
#include <cuda_runtime.h>
#include "selftest_cu.h"

__global__ void kScale(float* a, float s, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) a[i] = a[i] * s + 1.f;
}

// ---- streams, pinned host memory, asynchronous copies ---------------------------------------
static std::string tStreams() {
    const int n = 1000;
    const size_t bytes = n * sizeof(float);
    cudaStream_t s1, s2;
    ST_REQUIRE(cudaStreamCreate(&s1) == cudaSuccess && cudaStreamCreate(&s2) == cudaSuccess, "cudaStreamCreate failed");
    float *pinnedIn = 0, *pinnedOut = 0, *devA = 0, *devB = 0;
    ST_REQUIRE(cudaMallocHost((void**)&pinnedIn, bytes) == cudaSuccess, "cudaMallocHost failed");
    ST_REQUIRE(cudaHostAlloc((void**)&pinnedOut, 2 * bytes, cudaHostAllocDefault) == cudaSuccess, "cudaHostAlloc failed");
    cudaMalloc((void**)&devA, bytes);
    cudaMalloc((void**)&devB, bytes);
    for (int i = 0; i < n; ++i) pinnedIn[i] = (float)i;
    cudaMemcpyAsync(devA, pinnedIn, bytes, cudaMemcpyHostToDevice, s1);
    kScale<<<(n + 127) / 128, 128, 0, s1>>>(devA, 2.f, n);          // the stream is the 4th launch parameter
    cudaMemcpyAsync(pinnedOut, devA, bytes, cudaMemcpyDeviceToHost, s1);
    cudaMemsetAsync(devB, 0, bytes, s2);
    kScale<<<(n + 127) / 128, 128, 0, s2>>>(devB, 5.f, n);
    cudaMemcpyAsync(pinnedOut + n, devB, bytes, cudaMemcpyDeviceToHost, s2);
    ST_REQUIRE(cudaStreamSynchronize(s1) == cudaSuccess && cudaStreamSynchronize(s2) == cudaSuccess, "cudaStreamSynchronize failed");
    ST_LAUNCHED();
    std::string err;
    for (int i = 0; i < n && err.empty(); ++i) {
        if (pinnedOut[i] != 2.f * i + 1.f) err = st_fmt("stream 1: element %d is %g, expected %g", i, pinnedOut[i], 2.f * i + 1.f);
        else if (pinnedOut[n + i] != 1.f) err = st_fmt("stream 2: element %d is %g, expected 1", i, pinnedOut[n + i]);
    }
    cudaFree(devA); cudaFree(devB);
    cudaFreeHost(pinnedIn); cudaFreeHost(pinnedOut);
    if (cudaStreamDestroy(s1) != cudaSuccess || cudaStreamDestroy(s2) != cudaSuccess) return "cudaStreamDestroy failed";
    return err;
}

// ---- events ---------------------------------------------------------------------------------
static std::string tEvents() {
    const int n = 4096;
    cudaEvent_t start, stop;
    ST_REQUIRE(cudaEventCreate(&start) == cudaSuccess && cudaEventCreate(&stop) == cudaSuccess, "cudaEventCreate failed");
    StBuf<float> a(n, 1.f);
    cudaEventRecord(start);
    kScale<<<n / 256, 256>>>(a.dev, 3.f, n);
    cudaEventRecord(stop, 0);
    ST_REQUIRE(cudaEventSynchronize(stop) == cudaSuccess, "cudaEventSynchronize failed");
    float ms = -1.f;
    ST_REQUIRE(cudaEventElapsedTime(&ms, start, stop) == cudaSuccess, "cudaEventElapsedTime failed");
    ST_REQUIRE(ms >= 0.f, "elapsed time is %g ms", ms);     // a value, not a measurement of anything here
    ST_LAUNCHED();
    ST_REQUIRE(a.host()[n - 1] == 4.f, "kernel result is %g", a.host()[n - 1]);
    ST_REQUIRE(cudaEventDestroy(start) == cudaSuccess && cudaEventDestroy(stop) == cudaSuccess, "cudaEventDestroy failed");
    return "";
}

// ---- managed memory: one pointer used by the host and by a kernel -------------------------------
static std::string tManaged() {
    const int n = 512;
    float* shared = 0;
    ST_REQUIRE(cudaMallocManaged((void**)&shared, n * sizeof(float)) == cudaSuccess, "cudaMallocManaged failed");
    for (int i = 0; i < n; ++i) shared[i] = (float)(i % 10);
    kScale<<<n / 64, 64>>>(shared, 10.f, n);
    ST_REQUIRE(cudaDeviceSynchronize() == cudaSuccess, "cudaDeviceSynchronize failed");
    std::string err;
    for (int i = 0; i < n && err.empty(); ++i)
        if (shared[i] != 10.f * (i % 10) + 1.f) err = st_fmt("element %d is %g", i, shared[i]);
    cudaFree(shared);
    return err;
}

// ---- device properties, occupancy, more dynamic shared memory than the default -----------------
__global__ void kBigShared(int* out, int slots) {
    extern __shared__ int pool[];
    int t = threadIdx.x;
    pool[slots - 1 - t] = t;                    // the far end of the array
    __syncthreads();
    out[blockIdx.x * blockDim.x + t] = pool[slots - blockDim.x + t];
}
static std::string tProperties() {
    int count = 0;
    ST_REQUIRE(cudaGetDeviceCount(&count) == cudaSuccess && count >= 1, "cudaGetDeviceCount gave %d", count);
    cudaDeviceProp prop;
    ST_REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess, "cudaGetDeviceProperties failed");
    ST_REQUIRE(prop.warpSize == warpSize, "prop.warpSize is %d but warpSize is %d", prop.warpSize, warpSize);
    ST_REQUIRE(prop.maxThreadsPerBlock >= 64 && prop.sharedMemPerBlock >= 16384, "implausible limits");
    int coop = -1;
    ST_REQUIRE(cudaDeviceGetAttribute(&coop, cudaDevAttrCooperativeLaunch, 0) == cudaSuccess && coop == prop.cooperativeLaunch,
               "cudaDevAttrCooperativeLaunch is %d, prop.cooperativeLaunch is %d", coop, prop.cooperativeLaunch);
    int perSm = -1;
    ST_REQUIRE(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&perSm, kScale, 128, 0) == cudaSuccess && perSm >= 1,
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor gave %d", perSm);

    // Dynamic shared memory above the default needs an opt-in per kernel.
    size_t bytes = prop.sharedMemPerBlock + 4096;
    if (bytes > prop.sharedMemPerBlockOptin) return "";          // this device has no larger opt-in size
    ST_REQUIRE(cudaFuncSetAttribute(kBigShared, cudaFuncAttributeMaxDynamicSharedMemorySize, (int)bytes) == cudaSuccess,
               "cudaFuncSetAttribute failed");
    const int T = 64, slots = (int)(bytes / sizeof(int));
    StBuf<int> out(2 * T, -1);
    kBigShared<<<2, T, bytes>>>(out.dev, slots);
    ST_LAUNCHED();
    std::vector<int> o = out.host();
    for (int i = 0; i < 2 * T; ++i) ST_REQUIRE(o[i] == T - 1 - i % T, "element %d is %d", i, o[i]);
    return "";
}

// ---- a launch the device must refuse ----------------------------------------------------------
static std::string tRefused() {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    StBuf<float> a(16, 1.f);
    kScale<<<1, prop.maxThreadsPerBlock + 1>>>(a.dev, 2.f, 16);
    cudaError_t e = cudaGetLastError();
    ST_REQUIRE(e == cudaErrorInvalidConfiguration, "a block of %d threads gave: %s", prop.maxThreadsPerBlock + 1, cudaGetErrorString(e));
    ST_REQUIRE(cudaGetLastError() == cudaSuccess, "cudaGetLastError did not reset the error");
    ST_REQUIRE(a.host()[0] == 1.f, "the refused kernel ran anyway");
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"streams, cudaMemcpyAsync, cudaMemsetAsync, cudaMallocHost, cudaHostAlloc", tStreams},
    {"events", tEvents},
    {"cudaMallocManaged", tManaged},
    {"device properties, occupancy query, cudaFuncSetAttribute for dynamic shared memory", tProperties},
    {"an invalid launch configuration is refused", tRefused},
};
ST_END_CASES
