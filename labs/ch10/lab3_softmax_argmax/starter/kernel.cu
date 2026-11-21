// Lab 10.3 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M4: row-wise max, softmax and argmax over logits.
//
// Book: the two-stage warp-level reduction (section 10.8, Fig. 10.16, p. 243) on the warp
// reduce of Fig. 10.13 (p. 240), the coarsening loop of Fig. 10.20 (p. 247), and exercise 4's
// max operator (p. 250). Softmax and argmax are not in the book; you build them from those
// parts. Read README.md first: it states the contract, the tie rule and the tolerances.
//
// Contract:
//   x is [rows x cols] float32, row-major, batch first; rows >= 1, cols >= 1 (any value, not
//   only multiples of anything); every logit finite. One block of BLOCK_SIZE threads per row.
//   launchRowMax       out[r] = max over c of x[r][c]
//   launchSoftmaxRows  y[r][c] = exp(scale*x[r][c] - m) / sum over c' of exp(scale*x[r][c'] - m),
//                      where m = max over c of scale*x[r][c]
//   launchArgmaxRows   idx[r] = the LOWEST c with x[r][c] equal to the row's maximum
#include <cfloat>
#include <climits>

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // enough per-warp slots on every backend

// From lab 10.1: the shuffle mask has its own type (HIP requires 64 bits), and the warp
// width is warpSize, never a literal 32.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// TODO 1: warp level. Write a max and a sum version of lab 10.1's warp_reduce
//         (Fig. 10.13 with the typed mask and warpSize). A lane whose partner is past
//         the end of the warp gets its own value back: is that harmless for max?

// TODO 2: block level, the two stages of Fig. 10.16. Two things the book's kernel
//         does not need and you do:
//           - softmax needs the block's result in EVERY thread, not only thread 0;
//           - the stage-2 warp has more lanes than there are warp results (8 warps of
//             32 lanes, or 4 of 64), so its lanes without one must hold the identity.
//         Identities: 0 for the sum; -FLT_MAX for the max (finite on purpose: see README).

__global__ void rowMaxKernel(const float* x, float* out, int cols) {
    // TODO 3: block blockIdx.x reduces row blockIdx.x. Each thread first walks the row
    //         with a loop strided by blockDim.x (the coarsening loop of Fig. 10.20, run
    //         to the end of the row); then the block reduction; then one thread writes.
}

void launchRowMax(const float* x, float* out, int rows, int cols) {
    // TODO 4: one block of BLOCK_SIZE threads per row.
}

__global__ void softmaxRowsKernel(const float* x, float* y, int cols, float scale) {
    // TODO 5: three passes over the row:
    //           1. m = the max of scale * x[c]           (block reduction)
    //           2. s = the sum of expf(scale * x[c] - m)  (block reduction)
    //           3. y[c] = expf(scale * x[c] - m) / s
    //         Use expf, not __expf (README: tolerances). If both reductions use the same
    //         shared array, ask what the second one writes, and who may still be reading
    //         the first one's result.
}

void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale) {
    // TODO 6
}

__global__ void argmaxRowsKernel(const float* x, int* idx, int cols) {
    // TODO 7: reduce (value, index) pairs. Write ONE combine function and use it in the
    //         thread loop and in both warp stages; it must give the same pair whatever
    //         order the tree combines in, so decide what happens when two values are
    //         equal. A shuffle moves one value: the value and the index need one each.
    //         The identity pair must lose to every real pair, even to a logit of -FLT_MAX.
}

void launchArgmaxRows(const float* x, int* idx, int rows, int cols) {
    // TODO 8
}
