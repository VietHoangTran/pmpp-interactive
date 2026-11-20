// Lab X2.2 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M9, part 2: a 4-bit-weight matrix-vector product on llama.cpp's Q4_0 blocks,
//   Y[b][j] = sum over k of X[b][k] * w[j][k],   1 <= batch <= MAX_BATCH, in % 32 == 0,
// W [out x in/32] blocks of BlockQ4_0 (labs/x2/shared/qformat.h), X and Y float32, batch first.
//
// Not in the book. Module X2 is ours, anchored on appendix A section A.5 (p. 593) and
// chapter 15's intensity analysis. The block is llama.cpp's block_q4_0 exactly
// (notes/today/x2-facts.md D1):
//   struct { uint16_t d; uint8_t qs[16]; }, 18 bytes; d the FP16 bits of the scale; byte j
//   holds code j in its low nibble and code j + 16 in its high nibble; w = (code - 8) * d.
//
// Portability, not in the book: the lane mask has its own type (infra-facts B3), and the
// warp width is never a literal: warpSize in the kernel, the device property in the
// launcher. The 32 that appears is QK4_0, the block size.
#include <cstdint>
#include <cstdio>

#include "../../shared/qformat.h"   // BlockQ4_0, QK4_0, halfBitsToFloat

#define MAX_BATCH 8          // decode-sized, as lab 22.1
#define WARPS_PER_BLOCK 4    // 128 threads on a 32-lane warp, 256 on a 64-lane one

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Chapter 10's warp reduce, provided as in lab 22.1: every lane calls it, lane 0 gets the
// warp's sum. static, because the solutions of labs 22.1 and X2.1, linked beside this file
// for the benchmark, have their own.
static __device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(full, v, offset);
    }
    return v;
}

// The warp width of device 0, read once. Provided.
static int deviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

__global__ void gemvQ4_0Kernel(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out) {
    // TODO 1: one warp owns output j, as in lab 22.1's gather and lab X2.1.
    //   a. This thread's lane and its warp's output j; a warp past the end returns, all
    //      lanes together. Row j is in / QK4_0 blocks.
    //   b. Decide how the lanes share the blocks of row j: one lane per block, or several
    //      lanes per block (the X2.4 stepper shows the choice). Whatever you pick must work
    //      for every warp width, not only 32.
    //   c. Read the quants. sizeof(BlockQ4_0) is 18, so where does qs start in block 1? Can
    //      you read it as an int there? Bytes or 16-bit words are safe; see the README for
    //      why no emulator here would catch the unsafe choice.
    //   d. Unpack with masks and shifts. Which weights do the low and the high nibble of
    //      byte t stand for? (Not t and t + 1.)
    //   e. Convert d once per block with halfBitsToFloat. The -8 can be applied to every
    //      code, or once per block through the sum of the activations it meets
    //      (README, "Your part"); either is correct.
    //   f. Reduce each batch row's sum with warpSum (which lanes must call it?); lane 0
    //      writes Y[b][j].
}

void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out) {
    // TODO 2: one warp per output. Blocks of WARPS_PER_BLOCK warps of deviceWarpSize()
    //         threads. Return without launching when there is nothing to compute, and
    //         refuse (with a message) a batch above MAX_BATCH, an `in` that is not a
    //         multiple of QK4_0, or a warp width your mapping in TODO 1b cannot use.
}
