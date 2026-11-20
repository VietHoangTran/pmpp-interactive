// Lab X2.2 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M9, part 2: a 4-bit-weight matrix-vector product on llama.cpp's Q4_0 blocks,
//   Y[b][j] = sum over k of X[b][k] * w[j][k],   1 <= batch <= MAX_BATCH, in % 32 == 0,
// W [out x in/32] blocks of BlockQ4_0 (labs/x2/shared/qformat.h), X and Y float32, batch first.
//
// Not in the book. Module X2 is ours, anchored on appendix A section A.5 (p. 593) and
// chapter 15's intensity analysis. The block is llama.cpp's block_q4_0 exactly
// (ggml-common.h; notes/today/x2-facts.md D1):
//   struct { uint16_t d; uint8_t qs[16]; }, 18 bytes; d the FP16 bits of the scale; byte j
//   holds code j in its low nibble and code j + 16 in its high nibble; w = (code - 8) * d.
//
// The mapping (ours): lab 22.1's gather, one warp per output j, with LANES_PER_BLOCK = 4
// lanes sharing one block. Lane s of the 4 reads bytes 4s .. 4s + 3 of qs: their low nibbles
// are weights 4s .. 4s + 3 and their high nibbles weights 16 + 4s .. 16 + 4s + 3. A step of
// the warp covers warpSize / 4 consecutive blocks, warpSize / 4 x 18 contiguous bytes.
//   - qs is read as two 16-bit words, never as one int: an 18-byte block is only 2-byte
//     aligned, so in every other block qs starts at an address that is 2 mod 4 (A5).
//     llama.cpp's get_int_b2 does the same (A5).
//   - The -8 is folded out of the inner loop (E4): sum over the block of (q - 8) d x
//     = d (sum of q x - 8 sum of x). llama.cpp's CUDA path does it through Q8_1's s field
//     (D3), its Metal path and MLX through the activation sum (D4, D5).
//   - No integer dot product: the activations are float here (llama.cpp's CUDA path makes
//     them int8 to use dp4a, D3; __dp4a does not exist under HIP, B1).
//
// Portability, not in the book: the lane mask has its own type (infra-facts B3); the warp
// width is never a literal, and LANES_PER_BLOCK divides every warp width there is (32 on
// NVIDIA, 64 or 32 on AMD, ch05 facts B5). The 32 that appears is QK4_0, the block size.
#include <cstdint>
#include <cstdio>

#include "../../shared/qformat.h"   // BlockQ4_0, QK4_0, halfBitsToFloat

#define MAX_BATCH 8          // decode-sized, as lab 22.1
#define WARPS_PER_BLOCK 4    // 128 threads on a 32-lane warp, 256 on a 64-lane one
#define LANES_PER_BLOCK 4    // lanes sharing one Q4_0 block: 4 bytes of qs, 8 weights each

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Chapter 10's warp reduce, as in lab 22.1: every lane calls it, lane 0 gets the warp's sum.
// static, because the solutions of labs 22.1 and X2.1, linked beside this file for the
// benchmark, have their own.
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

__global__ void gemvQ4_0Kernel(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out) {
    const int lane = threadIdx.x % warpSize;
    const int j = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;   // this warp's output
    if (j >= out) {
        return;   // j is the same for every lane, so the whole warp leaves together
    }
    const int nb = in / QK4_0;                     // blocks in row j
    const BlockQ4_0* row = W + (size_t)j * nb;
    const int s = lane % LANES_PER_BLOCK;          // which 4 bytes of qs this lane reads
    const int blocksPerStep = warpSize / LANES_PER_BLOCK;

    float sum[MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }

    for (int i = lane / LANES_PER_BLOCK; i < nb; i += blocksPerStep) {
        const float d = halfBitsToFloat(row[i].d);
        // Bytes 4s .. 4s + 3 of qs as two 16-bit loads. qs is at offset 2 of a 2-byte
        // aligned block, so these are aligned; one 32-bit load would not be for every
        // other block (block g's qs starts at byte 18g + 2 of W: 2 mod 4 for even g).
        const uint16_t* qs16 = (const uint16_t*)row[i].qs;
        const uint32_t four = (uint32_t)qs16[2 * s] | ((uint32_t)qs16[2 * s + 1] << 16);
        float lo[4], hi[4];
        #pragma unroll
        for (int t = 0; t < 4; ++t) {
            lo[t] = (float)((four >> (8 * t)) & 0x0Fu);       // code of weight 4s + t
            hi[t] = (float)((four >> (8 * t + 4)) & 0x0Fu);   // code of weight 16 + 4s + t
        }
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                const float* x = X + (size_t)b * in + (size_t)i * QK4_0 + 4 * s;
                float qx = 0.0f, sx = 0.0f;
                #pragma unroll
                for (int t = 0; t < 4; ++t) {
                    qx += lo[t] * x[t] + hi[t] * x[16 + t];
                    sx += x[t] + x[16 + t];
                }
                sum[b] += d * (qx - 8.0f * sx);   // the -8 once per lane's share of the block
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

void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0 || in <= 0) {
        return;
    }
    if (batch > MAX_BATCH || in % QK4_0 != 0) {
        std::fprintf(stderr, "launchGemvQ4_0: need 1 <= batch <= %d and in a multiple of %d (got %d, %d)\n",
                     MAX_BATCH, QK4_0, batch, in);
        return;
    }
    const int width = deviceWarpSize();
    if (width <= 0 || width % LANES_PER_BLOCK != 0) {
        std::fprintf(stderr, "launchGemvQ4_0: warp width %d is unknown or not a multiple of %d\n", width,
                     LANES_PER_BLOCK);
        return;
    }
    // One warp per output: ceil(out / WARPS_PER_BLOCK) blocks of WARPS_PER_BLOCK warps.
    const int blocks = (out + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemvQ4_0Kernel<<<blocks, WARPS_PER_BLOCK * width>>>(X, W, Y, batch, in, out);
}
