// Labs X2.3 and X2.4, provided (do not edit). A STAND-IN for lab X2.2's reference solution,
// the fused float Q4_0 GEMV, with the signature labs X2.3 and X2.4 call:
//   void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out)
//   Y[batch x out] = X[batch x in] * W^T, W stored (out, in) as out * in / 32 Q4_0 blocks,
//   batch <= 8, in % 32 == 0.
// The labs' cmake files (../shared-g12/qpieces.cmake) build this file only when lab X2.2's
// solution is absent or has another signature; the driver prints which one it used.
//
// Not in the book, and ours. The shape is lab 22.1's gather (one warp per output, lanes over
// the row, chapter 10's shuffle reduce), with the dequantisation fused into the dot product
// and the -8 folded out of the inner loop through the sum of the activations (x2-facts E4):
//   sum over the block of (q - 8) d x  =  d (sum q x  -  8 sum x).
// Each lane takes whole blocks: 18 contiguous bytes, read as 16-bit words because the block
// is only 2-byte aligned (x2-facts A5).
//
// Portability: typed lane mask (HIP needs 64 bits, infra-facts B3); warp width from the
// device, never a literal 32.
#include <cstdint>
#include <cstdio>

#include "../shared-g12/qformats.h"

#define G12_Q4_MAX_BATCH 8
#define G12_Q4_WARPS_PER_BLOCK 4

#if defined(__HIPCC__)
typedef unsigned long long g12_q4_lane_mask_t;
#else
typedef unsigned int g12_q4_lane_mask_t;
#endif

// FP16 bits to float, exact (qref::halfToFloat on the device; no __half type is relied on).
static __device__ float g12Q4HalfToFloat(uint16_t h) {
    const unsigned sign = (unsigned)(h & 0x8000u) << 16;
    const unsigned e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f;
        return sign ? -v : v;
    }
    union { unsigned u; float f; } c;
    c.u = (e == 31) ? (sign | 0x7F800000u | (m << 13)) : (sign | ((e + 112u) << 23) | (m << 13));
    return c.f;
}

static __device__ float g12Q4WarpSum(float v) {
    const g12_q4_lane_mask_t full = ~(g12_q4_lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) v += __shfl_down_sync(full, v, offset);
    return v;
}

static int g12Q4DeviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

__global__ void gemvQ4_0G12Kernel(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out) {
    const int lane = threadIdx.x % warpSize;
    const int j = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;   // this warp's output
    if (j >= out) return;                                                          // the whole warp leaves
    const int nb = in / QK4_0;
    const BlockQ4_0* row = W + (size_t)j * nb;
    float sum[G12_Q4_MAX_BATCH];
    for (int b = 0; b < G12_Q4_MAX_BATCH; ++b) sum[b] = 0.0f;
    for (int kb = lane; kb < nb; kb += warpSize) {
        const float d = g12Q4HalfToFloat(row[kb].d);
        const uint16_t* q16 = (const uint16_t*)row[kb].qs;
        float acc[G12_Q4_MAX_BATCH], xs[G12_Q4_MAX_BATCH];
        for (int b = 0; b < G12_Q4_MAX_BATCH; ++b) { acc[b] = 0.0f; xs[b] = 0.0f; }
        for (int i = 0; i < 8; ++i) {
            const unsigned u = q16[i];                       // bytes 2i and 2i + 1 of qs
            const float c0 = (float)(u & 0xFu);              // weight 2i
            const float c1 = (float)((u >> 8) & 0xFu);       // weight 2i + 1
            const float c2 = (float)((u >> 4) & 0xFu);       // weight 2i + 16
            const float c3 = (float)((u >> 12) & 0xFu);      // weight 2i + 17
            for (int b = 0; b < G12_Q4_MAX_BATCH; ++b) {
                if (b < batch) {
                    const float* x = X + (size_t)b * in + (size_t)kb * QK4_0;
                    acc[b] += c0 * x[2 * i] + c1 * x[2 * i + 1] + c2 * x[2 * i + 16] + c3 * x[2 * i + 17];
                    xs[b] += x[2 * i] + x[2 * i + 1] + x[2 * i + 16] + x[2 * i + 17];
                }
            }
        }
        for (int b = 0; b < G12_Q4_MAX_BATCH; ++b) sum[b] += d * (acc[b] - 8.0f * xs[b]);
    }
    for (int b = 0; b < G12_Q4_MAX_BATCH; ++b) {
        const float t = (b < batch) ? g12Q4WarpSum(sum[b]) : 0.0f;   // `b < batch` is the same for every lane
        if (lane == 0 && b < batch) Y[(size_t)b * out + j] = t;
    }
}

void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0) return;
    if (batch > G12_Q4_MAX_BATCH || in % QK4_0 != 0) {
        std::fprintf(stderr, "launchGemvQ4_0: need batch <= %d and in %% 32 == 0 (got %d, %d)\n", G12_Q4_MAX_BATCH,
                     batch, in);
        return;
    }
    const int width = g12Q4DeviceWarpSize();
    if (width <= 0) {
        std::fprintf(stderr, "launchGemvQ4_0: could not read the device's warp width\n");
        return;
    }
    const int blocks = (out + G12_Q4_WARPS_PER_BLOCK - 1) / G12_Q4_WARPS_PER_BLOCK;
    gemvQ4_0G12Kernel<<<blocks, G12_Q4_WARPS_PER_BLOCK * width>>>(X, W, Y, batch, in, out);
}
