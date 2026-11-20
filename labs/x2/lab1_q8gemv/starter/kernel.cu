// Lab X2.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M9, part 1: a weight-only Q8_0 matrix-vector product for batch-1 decode,
//   Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k],   1 <= batch <= MAX_BATCH,
// with W stored (out, in) row-major as int8 codes Wq [out x in] and one FP16 scale per 32
// weights, held as uint16_t bits, in Wd [out x in/32]. X and Y are float32, batch first.
//
// Not in the book. Module X2 is ours, anchored on appendix A section A.5 (p. 593: fewer bytes
// per value raise compute intensity) and on chapter 15's intensity analysis.
//   - The quantisation rule and block size are llama.cpp's Q8_0 (notes/today/x2-facts.md D1).
//   - Codes and scales in two arrays is ours, as MLX keeps its scales apart (D5), so that
//     every row of codes starts 4-byte aligned (A5).
//   - The mapping is lab 22.1's gather: one warp per output, the lanes stride along the row
//     together, one register sum per batch row, a warp reduction, lane 0 writes.
//
// Portability, not in the book: the lane mask has its own type (infra-facts B3), and the
// warp width is never a literal: warpSize in the kernel, the device property in the
// launcher. The 32 that appears is QK8_0, the block size.
#include <cstdint>
#include <cstdio>

#include "../../shared/qformat.h"   // QK8_0, halfBitsToFloat

#define MAX_BATCH 8          // decode-sized, as lab 22.1
#define WARPS_PER_BLOCK 4    // 128 threads on a 32-lane warp, 256 on a 64-lane one
#define CODES_PER_WORD 4     // int8 codes in one 32-bit load

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Chapter 10's warp reduce, provided as in lab 22.1: every lane calls it, lane 0 gets the
// warp's sum. static, because lab 22.1's solution, linked beside this file for the
// benchmark, has its own.
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

// TODO 1: byte i (0..3) of a 32-bit word, as the signed code it holds. On a little-endian
//         machine, byte i of the word is the byte at address 4w + i. The byte is 0..255
//         once shifted down and masked; which of those values stand for negative codes, and
//         how do you turn them into -128..-1 without relying on how a compiler converts an
//         out-of-range value to int8_t?
static __device__ __forceinline__ float codeOf(uint32_t word, int i) {
    return 0.0f;   // a placeholder so the starter builds: replace it
}

__global__ void gemvQ8_0Kernel(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in,
                               int out) {
    // TODO 2: one warp owns output j, as in lab 22.1's gather.
    //   a. This thread's lane and its warp's output j. A warp past the end returns, all
    //      lanes together.
    //   b. Row j of the codes as 32-bit words: in / CODES_PER_WORD of them, starting at byte
    //      j x in. Why is that start 4-byte aligned here, and why would it not be in
    //      llama.cpp's interleaved block_q8_0?
    //   c. One running sum per batch row, MAX_BATCH floats, `b < batch` tested inside.
    //   d. The lanes stride along the words of row j together. Which block does word w
    //      belong to, and so which entry of row j's scales? Convert that scale once per word
    //      with halfBitsToFloat and multiply it in once for the word's four products.
    //   e. Reduce each sum with warpSum (which lanes must call it?), and lane 0 writes
    //      Y[b][j]. No atomics: nobody else writes that element.
}

void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in, int out) {
    // TODO 3: one warp per output. Blocks of WARPS_PER_BLOCK warps of deviceWarpSize()
    //         threads; how many blocks cover `out` outputs? Return without launching when
    //         there is nothing to compute, and refuse (with a message) a batch above
    //         MAX_BATCH or an `in` that is not a multiple of QK8_0.
}
