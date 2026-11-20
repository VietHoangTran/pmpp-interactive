// Lab 20.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
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
// Where this file differs from the page, and why (notes/today/ch20-facts.md unless named):
//   - p. 487 says gridDim.y and gridDim.z "are 0" for the book's 1D grid. Unused dimensions
//     are 1 (A1). Here grid y is used: it walks the batch.
//   - Fig. 20.4 reads S already scaled by 1/sqrt(d). Here the kernel applies `scale`, as lab
//     10.3's softmax does, so that S can come straight from the M3 linear layer (Q K^T,
//     unscaled). The book allows either place for the scale (p. 486).
//   - Fig. 20.4's mask is the loop bound idx <= blockIdx.x (square S, one sequence). Here
//     the bound is row + (nk - nq), the lower-right alignment FlashAttention (since 2.1) and
//     MLX use, so the same kernel serves a decode step (A8, D4). PyTorch's is_causal aligns
//     the other way when nq < nk (D5).
//   - NEG_INFINITY is not a CUDA name (A5). The identity of max is -FLT_MAX, finite, as in
//     lab 10.3: under Metal's fast math infinity is undefined (C5), and one contract serves
//     all three backends. Masked scores are never read, so no infinity is needed anywhere.
//   - cub::BlockReduce with cub::Max() and cub::Sum(): those functors were removed in CCCL 3.0
//     (infra A11) and hipCUB is spelt differently (B1). The block reductions are lab 10.3's
//     reference (M4), carried over unchanged below. They hand the result to every thread
//     through scratch[0], so the book's max_or_sum broadcast and its barriers (lines 12 to 14,
//     21 to 23) are inside them.
//   - The book writes the denominator D "for reuse during training" (p. 488). D alone cannot
//     rebuild P without the row max; FlashAttention-2 stores the logsumexp m + log(D) (A7),
//     and so does this kernel, as L. Lab 20.2's flash kernel must reproduce it.
//   - exp and log are written expf and logf: single precision, said explicitly (A5).
#include <cfloat>

// The course's CUDA front-end check (labs/pmpp.py) declares no CUDA Math API functions,
// so it is given the two device declarations used here. nvcc, hipcc and the emulators
// never see them.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
__device__ float logf(float x);
#endif

// ======================================================================================
// From lab 10.3's reference solution (milestone M4), unchanged: the two-stage block
// reduction of book section 10.8 (Fig. 10.16) on the warp reduction of Fig. 10.13, with a
// typed lane mask, warpSize instead of 32, and stage 2 guarded. See that lab's kernel.cu
// for where it differs from the page.
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
    const int row = blockIdx.x;                              // the book's blockIdx.x
    const size_t item = blockIdx.y;                          // which (sequence, head)
    const size_t rowStart = (item * nq + row) * (size_t)nk;  // first element of this row
    const float* S_row = S + rowStart;
    float* P_row = P + rowStart;

    // The mask is the loop bound, as in the book: columns 0 .. visible-1 are read, the rest
    // never are. With nq == nk this is the book's idx <= blockIdx.x.
    const int visible = row + (nk - nq) + 1;

    // Pass 1: m = largest scale*S of the visible columns (thread coarsening, then the block).
    float m = -FLT_MAX;
    for (int c = threadIdx.x; c < visible; c += blockDim.x) {
        m = maxOp(m, scale * S_row[c]);
    }
    m = blockReduceMax(m, scratch);

    // Every thread has just read m from scratch[0]; blockReduceSum writes scratch[0] again.
    // Without this barrier a fast warp 0 can overwrite m before a slow warp has read it
    // (lab 10.3's case; the book's line 19 plays the same role for its temp_store).
    __syncthreads();

    // Pass 2: s = sum of exp(scale*S - m). Every term is in (0, 1] and one of them is 1.
    float s = 0.0f;
    for (int c = threadIdx.x; c < visible; c += blockDim.x) {
        s += expf(scale * S_row[c] - m);
    }
    s = blockReduceSum(s, scratch);

    // Pass 3: every column of P is written, the masked ones with an explicit 0, and S is
    // read only where the mask allows it.
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < nk; c += blockDim.x) {
        if (c < visible) {
            P_row[c] = expf(scale * S_row[c] - m) * inv;
        } else {
            P_row[c] = 0.0f;
        }
    }

    // The logsumexp in place of the book's D (line 27).
    if (threadIdx.x == 0) {
        L[item * nq + row] = m + logf(s);
    }
}

void launchCausalSoftmax(const float* S, float* P, float* L, int batch, int nq, int nk, float scale) {
    if (batch <= 0 || nq <= 0) return;
    // x: one block per row (up to 2^31 - 1); y: one per item (up to 65535 on CUDA, ch05-facts A5).
    dim3 grid(nq, batch);
    causalSoftmaxKernel<<<grid, BLOCK_SIZE>>>(S, P, L, nq, nk, scale);
}
