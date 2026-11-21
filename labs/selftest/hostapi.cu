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

// ---- per-dimension limits: CUDA Programming Guide 13.5, Table 33 (the same for every listed
// compute capability): blocks at most 1024 x 1024 x 64, grids at most (2^31 - 1) x 65535 x 65535.
__global__ void kMark(int* out) {
    const int t = (threadIdx.z * blockDim.y + threadIdx.y) * blockDim.x + threadIdx.x;
    const int b = (blockIdx.z * gridDim.y + blockIdx.y) * gridDim.x + blockIdx.x;
    out[b * blockDim.x * blockDim.y * blockDim.z + t] = 1;
}
static std::string refusedLaunch(dim3 grid, dim3 block, const char* what) {
    StBuf<int> out(64, 0);
    kMark<<<grid, block>>>(out.dev);
    cudaError_t e = cudaGetLastError();
    ST_REQUIRE(e == cudaErrorInvalidConfiguration, "%s: expected cudaErrorInvalidConfiguration, got: %s", what, cudaGetErrorString(e));
    cudaDeviceSynchronize();
    for (int v : out.host()) ST_REQUIRE(v == 0, "%s: the refused kernel ran anyway", what);
#ifdef PMPP_EMU
    const std::string& msg = pmpp_emu::last_message();
    ST_REQUIRE(msg.find("Table 33") != std::string::npos, "%s: the emulator's message does not cite the guide's table: %s", what, msg.c_str());
#endif
    return "";
}
static std::string tDimensionLimits() {
    cudaDeviceProp prop;
    ST_REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess, "cudaGetDeviceProperties failed");
    ST_REQUIRE(prop.maxThreadsDim[0] == 1024 && prop.maxThreadsDim[1] == 1024 && prop.maxThreadsDim[2] == 64,
               "maxThreadsDim is %d x %d x %d", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    ST_REQUIRE(prop.maxGridSize[0] == 2147483647 && prop.maxGridSize[1] == 65535 && prop.maxGridSize[2] == 65535,
               "maxGridSize is %d x %d x %d", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    std::string r;
    if (!(r = refusedLaunch(dim3(1), dim3(1, 1, 65), "a block of 1 x 1 x 65 (65 threads, z over 64)")).empty()) return r;
    if (!(r = refusedLaunch(dim3(1), dim3(1, 1, 128), "a block of 1 x 1 x 128")).empty()) return r;
    if (!(r = refusedLaunch(dim3(1, 65536, 1), dim3(1), "a grid of 1 x 65536 x 1")).empty()) return r;
    if (!(r = refusedLaunch(dim3(1, 1, 65536), dim3(1), "a grid of 1 x 1 x 65536")).empty()) return r;
    if (!(r = refusedLaunch(dim3(2147483648u, 1, 1), dim3(1), "a grid of 2^31 x 1 x 1")).empty()) return r;
    ST_REQUIRE(cudaGetLastError() == cudaSuccess, "cudaGetLastError did not reset the error");
    // at the limits: a block of 1 x 1 x 64 and a block of 1 x 1024 x 1 run
    StBuf<int> out(1024, 0);
    kMark<<<1, dim3(1, 1, 64)>>>(out.dev);
    ST_LAUNCHED();
    kMark<<<1, dim3(1, 1024, 1)>>>(out.dev);
    ST_LAUNCHED();
    std::vector<int> o = out.host();
    for (int i = 0; i < 1024; ++i) ST_REQUIRE(o[i] == 1, "element %d was not written", i);
    return "";
}

// ---- device number, stream priorities, function attributes ----------------------------------------
static std::string tDeviceStreamFunc() {
    int dev = -1;
    ST_REQUIRE(cudaGetDevice(&dev) == cudaSuccess && dev == 0, "cudaGetDevice gave %d", dev);
    int least = 1, greatest = 1;
    ST_REQUIRE(cudaDeviceGetStreamPriorityRange(&least, &greatest) == cudaSuccess, "cudaDeviceGetStreamPriorityRange failed");
    ST_REQUIRE(greatest <= least, "greatest priority %d is above least %d (lower numbers are higher priorities)", greatest, least);
    ST_REQUIRE(cudaDeviceGetStreamPriorityRange(nullptr, &greatest) == cudaSuccess, "a NULL leastPriority was refused");
    cudaStream_t hi, lo;
    ST_REQUIRE(cudaStreamCreateWithPriority(&hi, cudaStreamNonBlocking, greatest - 100) == cudaSuccess &&
               cudaStreamCreateWithPriority(&lo, cudaStreamDefault, least + 100) == cudaSuccess, "cudaStreamCreateWithPriority failed");
    int pHi = 99, pLo = 99;
    ST_REQUIRE(cudaStreamGetPriority(hi, &pHi) == cudaSuccess && cudaStreamGetPriority(lo, &pLo) == cudaSuccess,
               "cudaStreamGetPriority failed");
    ST_REQUIRE(pHi == greatest && pLo == least, "out-of-range priorities were not clamped: got %d and %d, range %d..%d",
               pHi, pLo, greatest, least);
    StBuf<float> a(256, 1.f);
    kScale<<<2, 128, 0, hi>>>(a.dev, 2.f, 256);
    ST_REQUIRE(cudaStreamSynchronize(hi) == cudaSuccess, "cudaStreamSynchronize failed");   // the two launches share a.dev
    kScale<<<2, 128, 0, lo>>>(a.dev, 2.f, 256);
    ST_REQUIRE(cudaStreamSynchronize(lo) == cudaSuccess, "cudaStreamSynchronize failed");
    ST_LAUNCHED();
    ST_REQUIRE(a.host()[255] == 7.f, "two launches in priority streams gave %g", a.host()[255]);
    cudaStreamDestroy(hi);
    cudaStreamDestroy(lo);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    cudaFuncAttributes attr, attr2;
    ST_REQUIRE(cudaFuncGetAttributes(&attr, (const void*)kScale) == cudaSuccess, "cudaFuncGetAttributes (C form) failed");
    ST_REQUIRE(cudaFuncGetAttributes(&attr2, kScale) == cudaSuccess, "cudaFuncGetAttributes (template form) failed");
    ST_REQUIRE(attr.maxThreadsPerBlock >= 1 && attr.maxThreadsPerBlock <= prop.maxThreadsPerBlock &&
               attr2.maxThreadsPerBlock == attr.maxThreadsPerBlock, "maxThreadsPerBlock is %d and %d", attr.maxThreadsPerBlock,
               attr2.maxThreadsPerBlock);
#ifdef PMPP_EMU
    // Synthetic here: no device code exists, so there are no registers to count.
    ST_REQUIRE(attr.numRegs == 0 && attr.maxDynamicSharedSizeBytes == (int)prop.sharedMemPerBlock,
               "emulator: numRegs %d, maxDynamicSharedSizeBytes %d", attr.numRegs, attr.maxDynamicSharedSizeBytes);
#endif
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"per-dimension block and grid limits (Programming Guide Table 33) are enforced", tDimensionLimits},
    {"cudaGetDevice, stream priorities (clamped), cudaFuncGetAttributes", tDeviceStreamFunc},
    {"streams, cudaMemcpyAsync, cudaMemsetAsync, cudaMallocHost, cudaHostAlloc", tStreams},
    {"events", tEvents},
    {"cudaMallocManaged", tManaged},
    {"device properties, occupancy query, cudaFuncSetAttribute for dynamic shared memory", tProperties},
    {"an invalid launch configuration is refused", tRefused},
};
ST_END_CASES
