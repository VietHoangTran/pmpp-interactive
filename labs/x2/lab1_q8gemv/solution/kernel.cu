// Lab X2.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M9, part 1: a weight-only Q8_0 matrix-vector product for batch-1 decode,
//   Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k],   1 <= batch <= MAX_BATCH,
// with W stored (out, in) row-major as int8 codes Wq [out x in] and one FP16 scale per 32
// weights, held as uint16_t bits, in Wd [out x in/32]. X and Y are float32, batch first.
//
// Not in the book. Module X2 is ours, anchored on appendix A section A.5 (p. 593: fewer bytes
// per value raise compute intensity; notes/appA.md) and on chapter 15's intensity analysis.
//   - The quantisation rule and block size are llama.cpp's Q8_0 (quantize_row_q8_0_ref:
//     d = amax / 127, codes roundf(w / d), d stored as FP16; notes/today/x2-facts.md D1).
//   - Codes and scales in two arrays is ours, as MLX keeps its scales apart (D5): llama.cpp's
//     block_q8_0 interleaves a 2-byte scale with 32 codes, which would leave the codes
//     2-byte aligned (A5). Here every row of codes starts at a multiple of 32 bytes.
//   - The mapping is lab 22.1's gather: one warp per output j, the lanes stride along row j
//     together, one running sum per batch row in registers, chapter 10's warp reduction,
//     lane 0 writes. No atomics: nobody else writes Y[b][j].
//   - Each lane loads 4 codes as one 32-bit word, so a warp step reads warpSize x 4
//     consecutive bytes of row j, and multiplies the 4 products by their block's scale once.
//   - No __dp4a: it does not exist under HIP (x2-facts B1), and the activations are float
//     here, so the dot product is float multiply-adds on the converted codes. llama.cpp's
//     CUDA path quantises the activations to int8 to use dp4a instead (D3).
//
// Portability, not in the book: the lane mask has its own type (HIP's _sync intrinsics
// insist on 64 bits, infra-facts B3) and the warp width is never a literal: warpSize in the
// kernel, the device property in the launcher. The 32 that appears is QK8_0, the block size.
#include <cstdint>
#include <cstdio>

#include "../../shared/qformat.h"   // QK8_0, halfBitsToFloat

#define MAX_BATCH 8          // decode-sized, as lab 22.1; llama.cpp's mmvq stops at 8 too (ch22-facts D3)
#define WARPS_PER_BLOCK 4    // 128 threads on a 32-lane warp, 256 on a 64-lane one
#define CODES_PER_WORD 4     // int8 codes in one 32-bit load

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Chapter 10's warp reduce, as in lab 22.1: every lane calls it, lane 0 gets the warp's sum.
// static, because lab 22.1's solution, linked beside this file for the benchmark, has its own.
static __device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(full, v, offset);
    }
    return v;
}

// The warp width of device 0, read once.
static int deviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

// Byte i (0..3) of a word, as the signed code it holds: 0x00..0x7F are 0..127 and
// 0x80..0xFF are -128..-1. Byte i of the word is the byte at address 4w + i on a
// little-endian machine, which is what llama.cpp's get_int_b4 relies on too (ours).
static __device__ __forceinline__ float codeOf(uint32_t word, int i) {
    return (float)((int)(((word >> (8 * i)) & 0xFFu) ^ 0x80u) - 128);
}

__global__ void gemvQ8_0Kernel(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in,
                               int out) {
    const int lane = threadIdx.x % warpSize;
    const int j = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;   // this warp's output
    if (j >= out) {
        return;   // j is the same for every lane, so the whole warp leaves together
    }
    const int words = in / CODES_PER_WORD;                       // 32-bit words in row j
    // Row j of the codes starts at j x in bytes, a multiple of 32, and the allocation is
    // aligned, so the word loads are 4-byte aligned (x2-facts A5 is what goes wrong otherwise).
    const uint32_t* q = (const uint32_t*)(Wq + (size_t)j * in);
    const uint16_t* d = Wd + (size_t)j * (in / QK8_0);           // row j's scales

    float sum[MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }

    // Word w holds codes 4w .. 4w + 3, all in block w / 8. Each step of the warp reads
    // warpSize consecutive words of row j: warpSize x 4 bytes.
    for (int w = lane; w < words; w += warpSize) {
        const uint32_t four = q[w];
        const float scale = halfBitsToFloat(d[w / (QK8_0 / CODES_PER_WORD)]);
        const float q0 = codeOf(four, 0), q1 = codeOf(four, 1), q2 = codeOf(four, 2), q3 = codeOf(four, 3);
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                const float* x = X + (size_t)b * in + (size_t)w * CODES_PER_WORD;
                // The four products share one scale: multiply by it once.
                sum[b] += scale * (x[0] * q0 + x[1] * q1 + x[2] * q2 + x[3] * q3);
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
        // `b < batch` is the same for every lane, so every lane reaches the shuffles.
        const float total = (b < batch) ? warpSum(sum[b]) : 0.0f;
        if (lane == 0 && b < batch) {
            Y[(size_t)b * out + j] = total;   // nobody else writes this element
        }
    }
}

void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0 || in <= 0) {
        return;
    }
    if (batch > MAX_BATCH || in % QK8_0 != 0) {
        std::fprintf(stderr, "launchGemvQ8_0: need 1 <= batch <= %d and in a multiple of %d (got %d, %d)\n",
                     MAX_BATCH, QK8_0, batch, in);
        return;
    }
    const int width = deviceWarpSize();
    if (width <= 0) {
        std::fprintf(stderr, "launchGemvQ8_0: could not read the device's warp width\n");
        return;
    }
    // One warp per output: ceil(out / WARPS_PER_BLOCK) blocks of WARPS_PER_BLOCK warps.
    const int blocks = (out + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemvQ8_0Kernel<<<blocks, WARPS_PER_BLOCK * width>>>(X, Wq, Wd, Y, batch, in, out);
}
