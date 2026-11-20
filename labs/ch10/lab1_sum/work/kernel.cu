// Lab 10.1 starter (CUDA; also built for HIP and for the CPU emulators).
// One sum, three kernels. Book: sections 10.6 to 10.10, pp. 236-248.
//
//   (a) SharedMemorySumReductionKernel        Fig. 10.10, p. 237
//   (b) warp_reduce                            Fig. 10.13, p. 240
//       TwoStageWarpLevelSumReductionKernel    Fig. 10.16, p. 243
//   (c) CoarsenedSumReductionKernel            Figs. 10.18 and 10.20, pp. 245 and 247,
//                                              completed, plus exercise 5 (p. 250)
//
// Contract: rows of 2*blockDim.x floats for (a) and (b), one block per row, the sum of
// row b in out[b]. Any n >= 1 (below 2^31) for (c), the sum in *out. Block sizes are
// powers of two from 64 to 1024, chosen at run time by the test, so there is no
// BLOCK_DIM: use blockDim.x, and size shared arrays for the largest block.
//
// Where the book's listings will not do as printed (the lesson pages say why):
//   - Fig. 10.10 writes *output but has no `output` parameter.
//   - Fig. 10.13 passes the literal 0xffffffff and starts at WARP_SIZE/2 = 16.
//   - Fig. 10.16 lets all 32 lanes of warp 0 read partialSums_s[t], written or not.
//   - Fig. 10.18 never uses its N, and Fig. 10.20 stops at "...".

// cuda::atomic_ref for the commit in (c). On HIP, <cuda/atomic> comes from libhipcxx,
// which the HIP build links only where ROCm provides it (infra-facts B2), so this lab
// uses the legacy atomicAdd on HIP and builds either way: see TODO 6c.
#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents

// TODO 0: a type for the lane mask of __shfl_down_sync. CUDA's mask is an unsigned int.
//         HIP's _sync intrinsics refuse to compile unless the mask is a 64-bit integer,
//         in every warp width. __HIPCC__ is defined when HIP compiles this file.

// The book's helpers (pp. 238-239) use the constant WARP_SIZE = 32. Write them with
// the built-in warpSize instead, if you want them.

// ---------------------------------------------------------------- (a)
__global__ void SharedMemorySumReductionKernel(const float* input, float* output) {
    // TODO 1: block b reduces row b: input[2*blockDim.x*b ... 2*blockDim.x*(b+1) - 1].
    //   a. each thread adds its two elements, blockDim.x apart, into a shared array
    //   b. a tree in shared memory, stride halving from blockDim.x/2 down to 1, with
    //      one barrier per level; where does the barrier go so that step a is covered?
    //   c. one thread writes the row's sum to output[blockIdx.x]
}

void launchSharedMemorySum(const float* in, float* out, int rows, int block) {
    // TODO 2: one block of `block` threads per row.
}

// ---------------------------------------------------------------- (b)
// Every lane of a warp calls this with its value; lane 0 returns the warp's sum.
__device__ __inline__ float warp_reduce(float val) {
    // TODO 3: shuffle down with strides warpSize/2, ..., 2, 1, adding as you go.
    //         Use a full mask of your type from TODO 0, not a literal.
    return val;
}

__global__ void TwoStageWarpLevelSumReductionKernel(const float* input, float* output) {
    // TODO 4: block b reduces row b.
    //   a. stage 1: each thread adds its two elements in a register; every warp runs
    //      warp_reduce
    //   b. lane 0 of each warp stores its sum in a shared array indexed by warp. How
    //      many entries does the array need for any block size on any warp width?
    //   c. one barrier
    //   d. stage 2: warp 0 loads one per-warp sum per lane and runs warp_reduce again.
    //      Lanes with no warp behind them must load something harmless: what?
    //   e. thread 0 writes output[blockIdx.x]
}

void launchTwoStageWarpSum(const float* in, float* out, int rows, int block) {
    // TODO 5: one block of `block` threads per row.
}

// ---------------------------------------------------------------- (c)
__global__ void CoarsenedSumReductionKernel(const float* input, float* output,
                                            unsigned int N, unsigned int coarse) {
    // TODO 6: block b owns the segment of 2*coarse*blockDim.x elements starting at
    //         2*coarse*blockDim.x*b.
    //   a. each thread adds its 2*coarse elements of the segment serially, blockDim.x
    //      apart (so the block reads contiguous runs); an element at or past N adds 0
    //   b. the same two-stage block reduction as (b). Every thread takes part, even one
    //      whose elements were all out of range: why?
    //   c. thread 0 adds the block's sum into *output with ONE atomic, as Fig. 10.18:
    //      cuda::atomic_ref<float, cuda::thread_scope_device> and fetch_add with
    //      cuda::memory_order_relaxed. When __HIPCC__ is defined, use the legacy
    //      atomicAdd(output, value) instead: also relaxed and device-wide
    //      (infra-facts A1, B1).
}

void launchCoarsenedSum(const float* in, float* out, int n, int block, int coarse) {
    // TODO 7: the blocks add into *out, so it must hold 0 first, on every call.
    //         Then launch enough blocks of `block` threads to cover all n elements.
}
