// Lab 20.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M6, part one: the causal mask and softmax of attention, with the logsumexp.
// Book: section 20.3, Fig. 20.4 (pp. 486-488).
//
// Contract (README.md states it in full):
//   S and P are [batch x nq x nk] float32, row-major, batch first (batch = sequences x heads);
//   L is [batch x nq]. 1 <= nq <= nk. Row r of item b sees the columns c <= r + (nk - nq):
//   the mask is aligned to the lower right, so nq == nk is the book's square case and nq == 1
//   is one decode step that sees every key. With m the largest scale*S over the visible
//   columns of a row and s = sum of exp(scale*S - m) over them:
//     P = exp(scale*S - m) / s on the visible columns, exactly 0 on the masked ones;
//     L = m + log(s), the logsumexp of the visible scaled scores.
//   One block of BLOCK_SIZE threads per row; grid x = rows (nq), grid y = items (batch).
//
// The page's Fig. 20.4 needs four corrections; README.md lists them. In short: no
// NEG_INFINITY (use -FLT_MAX), no CUB functors (the M4 reductions below replace them), and
// L in place of the book's D.
#include <cfloat>

// The course's CUDA front-end check declares no CUDA Math API functions.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
__device__ float logf(float x);
#endif

// ======================================================================================
// From lab 10.3's reference solution (milestone M4), unchanged: the two-stage block
// reduction of book section 10.8 (Fig. 10.16) on the warp reduction of Fig. 10.13, with a
// typed lane mask, warpSize instead of 32, and stage 2 guarded. Use it; do not edit it.
// ======================================================================================
#define BLOCK_SIZE 256                          // threads per block; one block per row
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has (ch10-facts B2)
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots in shared memory

// Fig. 10.11's first parameter. CUDA: unsigned. HIP: every _sync intrinsic static_asserts
// a 64-bit mask (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// Finite inputs only. On NaN a comparison-based max and CUDA's fmaxf disagree
// (ch10-facts A10); NaN is outside the contract.
__device__ float maxOp(float a, float b) { return a > b ? a : b; }

// Every lane calls these; lane 0 returns the warp's result.
__device__ float warpReduceMax(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val = maxOp(val, __shfl_down_sync(full, val, stride));
    }
    return val;
}

__device__ float warpReduceSum(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

// Called by every thread of the block. Returns the block's result to every thread.
// scratch: MAX_WARPS floats of shared memory.
__device__ float blockReduceMax(float val, float* scratch) {
    val = warpReduceMax(val);                          // stage 1: every warp, in registers
    if (laneIdx() == 0) {
        scratch[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {                              // stage 2: warp 0 reduces the warp results
        val = laneIdx() < warpsInBlock() ? scratch[laneIdx()] : -FLT_MAX;
        val = warpReduceMax(val);
        if (laneIdx() == 0) {
            scratch[0] = val;                          // hand the result to the whole block
        }
    }
    __syncthreads();
    return scratch[0];
}

__device__ float blockReduceSum(float val, float* scratch) {
    val = warpReduceSum(val);
    if (laneIdx() == 0) {
        scratch[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        val = laneIdx() < warpsInBlock() ? scratch[laneIdx()] : 0.0f;
        val = warpReduceSum(val);
        if (laneIdx() == 0) {
            scratch[0] = val;
        }
    }
    __syncthreads();
    return scratch[0];
}
// ======================================================================================
// End of the M4 code from lab 10.3.
// ======================================================================================

// Fig. 20.4, for a batch of rows. Block (row, item) owns row `row` of item `item`.
__global__ void causalSoftmaxKernel(const float* S, float* P, float* L, int nq, int nk, float scale) {
    __shared__ float scratch[MAX_WARPS];

    // TODO 1: which row and which item does this block own, and where does that row start
    //         in S and P? (Items are nq x nk each, one after another.) Compute in size_t:
    //         a 2048 x 2048 batch of 3 already has more than 12 million elements.

    // TODO 2: how many columns of this row are visible? With nq == nk it must give the
    //         book's bound (column <= row); with nq == 1 it must give every column.

    // TODO 3: pass 1. Each thread walks the visible columns of the row, blockDim.x apart,
    //         keeping the largest scale * S it sees; then blockReduceMax gives the row's m
    //         to every thread. The mask is the loop bound: a masked score is never read.

    // TODO 4: blockReduceMax and blockReduceSum share `scratch`. What has to happen
    //         between the two calls? (Lab 10.3 met the same question.)

    // TODO 5: pass 2. The sum of exp(scale * S - m) over the visible columns, then
    //         blockReduceSum.

    // TODO 6: pass 3. Write every column of P: the softmax on the visible ones, exactly 0
    //         on the others. Do not read S where the mask says no.

    // TODO 7: one thread writes L for this row: m + log(sum).
}

void launchCausalSoftmax(const float* S, float* P, float* L, int batch, int nq, int nk, float scale) {
    // TODO 8: one block of BLOCK_SIZE threads per row of every item: grid x covers the
    //         rows, grid y the items.
}
