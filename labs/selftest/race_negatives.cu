// Self-test, race flavour only: three kernels with a real data race each. Every
// case must be REPORTED by ThreadSanitizer (whatever values the run produced).
#include "selftest_cu.h"

// 1. A missing barrier: threads read shared slots their neighbours may not have written yet.
__global__ void kNoBarrier(const int* in, int* out) {
    __shared__ int stage[64];
    int t = threadIdx.x, g = blockIdx.x * blockDim.x + t;
    stage[t] = in[g];
    // BUG: no __syncthreads() here
    out[g] = stage[63 - t];
}
static std::string tNoBarrier() {
    StBuf<int> in(std::vector<int>(128, 3)), out(128, 0);
    kNoBarrier<<<2, 64>>>(in.dev, out.dev);
    ST_LAUNCHED();
    return "";
}

// 2. Blocks racing with each other: every block adds its sum to one global total.
__global__ void kBlocksRace(const int* in, int* total) {
    __shared__ int partial[32];
    int t = threadIdx.x;
    partial[t] = in[blockIdx.x * blockDim.x + t];
    __syncthreads();
    if (t == 0) {
        int s = 0;
        for (int i = 0; i < 32; ++i) s += partial[i];
        *total += s;                            // BUG: needs atomicAdd, blocks run at the same time
    }
}
static std::string tBlocksRace() {
    StBuf<int> in(std::vector<int>(16 * 32, 1)), total(1, 0);
    kBlocksRace<<<16, 32>>>(in.dev, total.dev);
    ST_LAUNCHED();
    return "";
}

// 3. A warp shuffle is not a memory barrier (CUDA guide: "These intrinsics do
//    not imply a memory barrier"); only __syncwarp orders memory accesses.
__global__ void kShuffleIsNoBarrier(int* out) {
    __shared__ int stage[32];
    int t = threadIdx.x;
    stage[t] = t;
    int moved = __shfl_xor_sync(0xffffffffu, t, 1);
    out[t] = stage[t ^ 1] + moved;              // BUG: needs __syncwarp() before reading a neighbour's slot
}
static std::string tShuffleIsNoBarrier() {
    StBuf<int> out(32, 0);
    kShuffleIsNoBarrier<<<1, 32>>>(out.dev);
    ST_LAUNCHED();
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"missing __syncthreads between writing and reading shared memory", tNoBarrier},
    {"blocks adding to one global without an atomic", tBlocksRace},
    {"shared memory read after a shuffle instead of after __syncwarp", tShuffleIsNoBarrier},
};
ST_END_CASES
