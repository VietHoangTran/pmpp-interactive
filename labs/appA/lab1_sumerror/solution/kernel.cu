// Lab A.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Summation error: order, accumulator width and compensation.
//
// Book: section A.6 (pp. 593-594). A float sum depends on the order of its additions; a
// reduction tree adds in a different order from a loop, and can be more or less accurate;
// for something sturdier the book points to Kahan's compensated summation (ref. [5]) without
// describing it. Appendix A shows no code. The tree here is chapter 10's shared-memory
// reduction (Fig. 10.10, p. 237) behind a coarsening loop (Fig. 10.20, p. 247). The templated
// accumulator, the compensated loop and the 16-bit accumulators are ours.
//
// Contract (README.md states it in full): x holds n >= 1 float32 values; out[0] receives their
// sum, computed in the accumulator type AccT and converted to float. block is a power of two
// from 1 to MAX_BLOCK, coarse >= 1. partials holds at least ceil(n / block) floats.
//   (a) sumSerial  one thread adds x[0], x[1], ... in index order: the CPU loop's order.
//   (b) sumTree    thread t of block b adds x[b*B*coarse + t + j*B], j = 0 .. coarse-1, in that
//                  order (B = blockDim.x); the block combines its B sums with the tree; a second
//                  launch of the same kernel reduces the per-block sums.
//   (c) sumKahan   (b) with a compensated (Kahan) loop in each thread. The tree stays plain.
//
// Where this differs from Fig. 10.10:
//   - The tree combines blockDim.x per-thread sums, not 2*blockDim.x inputs: the coarsening
//     loop takes the place of Fig. 10.10's add-on-load.
//   - The book's BLOCK_DIM (compile time) is blockDim.x (run time); shared memory is sized for
//     the largest block.
//   - Fig. 10.10 writes *output but has no output parameter; out is restored, as in lab 10.1.
//   - Fig. 10.20 ends with an atomicAdd per block, which adds the block sums in whatever order
//     the blocks finish. Here they go to `partials` and a second launch adds them, so every run
//     adds in the same order and the CPU can model it exactly.
#include "../sumerror.h"

#define MAX_BLOCK 1024   // the largest block these kernels accept

// ---------------------------------------------------------------- additions kept as written
// The compensation c = (t - sum) - y is 0 in real arithmetic, so a compiler that may
// reassociate could delete it (notes/today/appA-facts.md A13, C3). For float, __fadd_rn and
// __fsub_rn are the additions NVIDIA documents as never merged into a fused multiply-add
// (A9); whether nvcc ever reassociates is not documented, and the tests show it. HIP-Clang
// does not reassociate by default (B5). The 16-bit types use their documented operators
// (A16, B1).
__device__ __forceinline__ float addRn(float a, float b) { return __fadd_rn(a, b); }
__device__ __forceinline__ float subRn(float a, float b) { return __fsub_rn(a, b); }
#if PMPP_NARROW_ACC
__device__ __forceinline__ __half addRn(__half a, __half b) { return a + b; }
__device__ __forceinline__ __half subRn(__half a, __half b) { return a - b; }
__device__ __forceinline__ __nv_bfloat16 addRn(__nv_bfloat16 a, __nv_bfloat16 b) { return a + b; }
__device__ __forceinline__ __nv_bfloat16 subRn(__nv_bfloat16 a, __nv_bfloat16 b) { return a - b; }
#endif

// ---------------------------------------------------------------- (a)
template <typename AccT>
__global__ void sumSerial(const float* x, float* out, int n) {
    AccT sum = AccT(0.0f);
    for (int i = 0; i < n; ++i) {
        sum = sum + AccT(x[i]);          // one rounding to AccT per addition
    }
    out[0] = float(sum);
}

// ---------------------------------------------------------------- the block's tree
// Fig. 10.10's tree on the block's per-thread sums: at every level thread t adds the value
// `stride` places to its right, and stride halves. The barrier at the top of each level also
// orders the stores of the first one. Thread 0 writes the block's sum.
template <typename AccT>
__device__ void treeToOutput(AccT* partial_s, AccT value, float* out) {
    unsigned int t = threadIdx.x;
    partial_s[t] = value;
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            partial_s[t] = partial_s[t] + partial_s[t + stride];
        }
    }
    if (t == 0) {
        out[blockIdx.x] = float(partial_s[0]);   // thread 0 made the last write to partial_s[0]
    }
}

// ---------------------------------------------------------------- (b)
template <typename AccT>
__global__ void sumTree(const float* x, float* out, int n, int coarse) {
    __shared__ AccT partial_s[MAX_BLOCK];
    // This thread's `coarse` elements, blockDim.x apart: at each step j the threads of a warp
    // read neighbouring floats, so the loads are coalesced (chapter 6).
    unsigned int i = blockIdx.x * blockDim.x * coarse + threadIdx.x;
    AccT sum = AccT(0.0f);
    for (int j = 0; j < coarse; ++j, i += blockDim.x) {
        if (i < (unsigned int)n) {
            sum = sum + AccT(x[i]);
        }
    }
    treeToOutput(partial_s, sum, out);   // every thread takes part, in range or not
}

// ---------------------------------------------------------------- (c)
template <typename AccT>
__global__ void sumKahan(const float* x, float* out, int n, int coarse) {
    __shared__ AccT partial_s[MAX_BLOCK];
    unsigned int i = blockIdx.x * blockDim.x * coarse + threadIdx.x;
    AccT sum = AccT(0.0f);
    AccT c = AccT(0.0f);                  // minus what the last addition lost
    for (int j = 0; j < coarse; ++j, i += blockDim.x) {
        if (i < (unsigned int)n) {
            AccT y = subRn(AccT(x[i]), c);      // the element plus what was lost before
            AccT t = addRn(sum, y);             // big + small: low bits of y fall off here
            c = subRn(subRn(t, sum), y);        // (what was really added) - (what we meant to add)
            sum = t;
        }
    }
    treeToOutput(partial_s, sum, out);
}

// ---------------------------------------------------------------- launchers
static bool shapeOk(int n, int coarse, int block) {
    return n >= 1 && coarse >= 1 && block >= 1 && block <= MAX_BLOCK && (block & (block - 1)) == 0;
}

// Pass 1: one partial sum per block. Pass 2: one block reduces the `blocks` partials, each
// thread first adding ceil(blocks / block) of them. Both passes launch the same kernel.
template <typename AccT>
static void treeTwoPasses(const float* x, float* partials, float* out, int n, int coarse, int block) {
    int blocks = (n + block * coarse - 1) / (block * coarse);
    sumTree<AccT><<<blocks, block>>>(x, partials, n, coarse);
    int coarse2 = (blocks + block - 1) / block;
    sumTree<AccT><<<1, block>>>(partials, out, blocks, coarse2);
}

template <typename AccT>
static void kahanTwoPasses(const float* x, float* partials, float* out, int n, int coarse, int block) {
    int blocks = (n + block * coarse - 1) / (block * coarse);
    sumKahan<AccT><<<blocks, block>>>(x, partials, n, coarse);
    int coarse2 = (blocks + block - 1) / block;
    sumKahan<AccT><<<1, block>>>(partials, out, blocks, coarse2);
}

bool launchSumSerial(const float* x, float* out, int n, Acc acc) {
    if (n < 1) return false;
    switch (acc) {
    case AccFloat: sumSerial<float><<<1, 1>>>(x, out, n); return true;
#if PMPP_NARROW_ACC
    case AccHalf: sumSerial<__half><<<1, 1>>>(x, out, n); return true;
    case AccBfloat: sumSerial<__nv_bfloat16><<<1, 1>>>(x, out, n); return true;
#endif
    default: return false;
    }
}

bool launchSumTree(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc) {
    if (!shapeOk(n, coarse, block)) return false;
    switch (acc) {
    case AccFloat: treeTwoPasses<float>(x, partials, out, n, coarse, block); return true;
#if PMPP_NARROW_ACC
    case AccHalf: treeTwoPasses<__half>(x, partials, out, n, coarse, block); return true;
    case AccBfloat: treeTwoPasses<__nv_bfloat16>(x, partials, out, n, coarse, block); return true;
#endif
    default: return false;
    }
}

bool launchSumKahan(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc) {
    if (!shapeOk(n, coarse, block)) return false;
    switch (acc) {
    case AccFloat: kahanTwoPasses<float>(x, partials, out, n, coarse, block); return true;
#if PMPP_NARROW_ACC
    case AccHalf: kahanTwoPasses<__half>(x, partials, out, n, coarse, block); return true;
    case AccBfloat: kahanTwoPasses<__nv_bfloat16>(x, partials, out, n, coarse, block); return true;
#endif
    default: return false;
    }
}
