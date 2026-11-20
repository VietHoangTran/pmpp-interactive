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

// 4. A race between two barriers: each thread reads its neighbour's slot, then overwrites
//    its own slot before the next barrier, while the neighbour may still be reading it.
//    Every barrier of a block once shared one happens-before token, so a thread leaving
//    barrier 1 late could "synchronise" with a neighbour already at barrier 2 and hide
//    this race; the tokens now alternate per barrier.
__global__ void kBetweenBarriers(int* out) {
    __shared__ int s[64];
    int t = threadIdx.x;
    s[t] = t;
    __syncthreads();
    int v = s[(t + 1) % 64];
    s[t] = v + 1;                               // BUG: needs a __syncthreads() between the read and this write
    __syncthreads();
    out[t] = s[t];
}
static std::string tBetweenBarriers() {
    StBuf<int> out(64, 0);
    kBetweenBarriers<<<1, 64>>>(out.dev);
    ST_LAUNCHED();
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"missing __syncthreads between writing and reading shared memory", tNoBarrier},
    {"blocks adding to one global without an atomic", tBlocksRace},
    {"shared memory read after a shuffle instead of after __syncwarp", tShuffleIsNoBarrier},
    {"a neighbour's slot overwritten between two barriers", tBetweenBarriers},
};
ST_END_CASES
