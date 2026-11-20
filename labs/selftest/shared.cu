// Self-test: __shared__ variables, dynamic shared memory, barriers.
#include "selftest_cu.h"

// ---- several declarations, 2-D arrays, a scalar, a helper with its own shared array --------
__device__ int reversedNeighbour(int v) {
    __shared__ int helper[64];                 // shared by the block, declared in a __device__ function
    int t = threadIdx.x;
    helper[t] = v;
    __syncthreads();
    int r = helper[blockDim.x - 1 - t];
    __syncthreads();
    return r;
}
__global__ void kShared(const int* in, int* out, int* blockSums) {
    __shared__ int tile[4][16], total;
    __shared__ float weights[64];
    int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
    tile[t / 16][t % 16] = in[g];
    weights[t] = 0.5f * t;
    if (t == 0) total = 0;
    __syncthreads();
    // every thread reads a slot written by another thread
    int a = tile[(t / 16 + 1) % 4][t % 16];
    float w = weights[(t + 7) % 64];
    __syncthreads();
    if (t == 0) { int s = 0; for (int i = 0; i < 64; ++i) s += tile[i / 16][i % 16]; total = s; }
    __syncthreads();
    out[g] = a + (int)(2.f * w) + reversedNeighbour(in[g]);
    blockSums[g] = total;
}
static std::string tShared() {
    const int blocks = 5, T = 64, n = blocks * T;
    std::vector<int> h(n);
    for (int i = 0; i < n; ++i) h[i] = (i * 31) % 97;
    StBuf<int> in(h), out(n, -1), sums(n, -1);
    kShared<<<blocks, T>>>(in.dev, out.dev, sums.dev);
    ST_LAUNCHED();
    std::vector<int> o = out.host(), s = sums.host();
    for (int b = 0; b < blocks; ++b) {
        int total = 0;
        for (int t = 0; t < T; ++t) total += h[b * T + t];
        for (int t = 0; t < T; ++t) {
            int a = h[b * T + ((t / 16 + 1) % 4) * 16 + t % 16];
            int want = a + (t + 7) % 64 + h[b * T + T - 1 - t];
            ST_REQUIRE(o[b * T + t] == want, "block %d thread %d: got %d, expected %d", b, t, o[b * T + t], want);
            ST_REQUIRE(s[b * T + t] == total, "block %d thread %d saw block total %d, expected %d", b, t, s[b * T + t], total);
        }
    }
    return "";
}

// ---- dynamic shared memory, one array carved in two ------------------------------------------
__global__ void kDynamic(const float* in, float* out, int half) {
    extern __shared__ float pool[];
    float* lo = pool;
    float* hi = pool + half;
    int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
    lo[t] = in[g];
    hi[t] = -in[g];
    __syncthreads();
    out[g] = lo[(t + 1) % half] + 2.f * hi[(t + 2) % half];
}
static std::string tDynamic() {
    const int blocks = 3, T = 96, n = blocks * T;
    std::vector<float> h(n);
    for (int i = 0; i < n; ++i) h[i] = (float)((i * 7) % 23);
    StBuf<float> in(h), out(n, -1.f);
    kDynamic<<<blocks, T, 2 * T * sizeof(float)>>>(in.dev, out.dev, T);
    ST_LAUNCHED();
    std::vector<float> o = out.host();
    for (int b = 0; b < blocks; ++b)
        for (int t = 0; t < T; ++t) {
            float want = h[b * T + (t + 1) % T] - 2.f * h[b * T + (t + 2) % T];
            ST_REQUIRE(o[b * T + t] == want, "block %d thread %d: got %g, expected %g", b, t, o[b * T + t], want);
        }
    return "";
}

// ---- __syncthreads_count / _and / _or ------------------------------------------------------------
__global__ void kSyncVotes(int* count, int* all, int* any) {
    int t = threadIdx.x;
    count[t] = __syncthreads_count(t % 4 == 0);
    int allTrue = __syncthreads_and(t < 1000);
    int allButOne = __syncthreads_and(t != 3);
    int oneTrue = __syncthreads_or(t == 17);
    int noneTrue = __syncthreads_or(t > 1000);
    all[t] = (allTrue != 0) * 10 + (allButOne != 0);
    any[t] = (oneTrue != 0) * 10 + (noneTrue != 0);
}
static std::string tSyncVotes() {
    const int T = 72;
    StBuf<int> count(T, -1), all(T, -1), any(T, -1);
    kSyncVotes<<<1, T>>>(count.dev, all.dev, any.dev);
    ST_LAUNCHED();
    std::vector<int> c = count.host(), a = all.host(), o = any.host();
    for (int t = 0; t < T; ++t) {
        ST_REQUIRE(c[t] == T / 4, "__syncthreads_count in thread %d gave %d, expected %d", t, c[t], T / 4);
        ST_REQUIRE(a[t] == 10, "__syncthreads_and in thread %d gave %d (10 = true then false)", t, a[t]);
        ST_REQUIRE(o[t] == 10, "__syncthreads_or in thread %d gave %d (10 = true then false)", t, o[t]);
    }
    return "";
}

#ifdef PMPP_EMU
// ---- emulator only ---------------------------------------------------------------------------------
// A block must not see what the previous block left in shared memory: the
// emulator hands every block fresh storage filled with 0xCD.
__global__ void kStale(unsigned* out) {
    __shared__ unsigned leftover[32];
    out[blockIdx.x * blockDim.x + threadIdx.x] = leftover[threadIdx.x];    // read before anyone wrote
    __syncthreads();
    leftover[threadIdx.x] = 42u;
}
static std::string tStale() {
    StBuf<unsigned> out(4 * 32, 0u);
    kStale<<<4, 32>>>(out.dev);
    ST_LAUNCHED();
    std::vector<unsigned> o = out.host();
    for (int i = 0; i < 4 * 32; ++i) ST_REQUIRE(o[i] == 0xCDCDCDCDu, "element %d read 0x%08x from shared memory nobody wrote", i, o[i]);
    return "";
}

// Half of the block waits at a barrier the other half never reaches... and the other half
// waits at a second barrier. On the emulator the two barriers pair up; a third one cannot.
__global__ void kStuck(int* out) {
    if (threadIdx.x < 8) { __syncwarp(0xffu); __syncthreads(); out[threadIdx.x] = 1; }
    else if (threadIdx.x < 16) { __syncwarp(0xffffu); }      // names lanes 0..7, which are at the barrier
}
static std::string tStuck() {
    StBuf<int> out(16, 0);
    kStuck<<<1, 16>>>(out.dev);
    cudaError_t e = cudaGetLastError();
    ST_REQUIRE(e == cudaErrorLaunchFailure, "expected the emulator to stop the kernel, got: %s", cudaGetErrorString(e));
    ST_REQUIRE(pmpp_emu::last_message().find("deadlock") != std::string::npos, "message: %s", pmpp_emu::last_message().c_str());
    // the emulator must be usable again afterwards
    StBuf<int> in(std::vector<int>(64, 1)), o2(64, 0), s2(64, 0);
    kShared<<<1, 64>>>(in.dev, o2.dev, s2.dev);
    ST_LAUNCHED();
    ST_REQUIRE(s2.host()[5] == 64, "a launch after a stopped one gave %d", s2.host()[5]);
    return "";
}
#endif

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"__shared__: several declarations, 2-D, scalar, in a __device__ function, one copy per block", tShared},
    {"dynamic shared memory (extern __shared__, third launch parameter)", tDynamic},
    {"__syncthreads_count, __syncthreads_and, __syncthreads_or", tSyncVotes},
#ifdef PMPP_EMU
    {"emulator: every block starts with unwritten (0xCD) shared memory", tStale},
    {"emulator stops a kernel whose threads can never meet, then keeps working", tStuck},
#endif
};
ST_END_CASES
