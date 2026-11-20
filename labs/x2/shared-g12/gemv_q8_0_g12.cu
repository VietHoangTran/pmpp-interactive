// Lab X2.4, provided (do not edit). A STAND-IN for lab X2.1's reference solution, the fused
// float Q8_0 GEMV, with the signature lab X2.4 calls:
//   void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y,
//                       int batch, int in, int out)
//   Y[batch x out] = X[batch x in] * W^T, W stored (out, in): Wq holds out * in int8 codes,
//   row-major; Wd holds out * in / 32 FP16 scales, one per 32 codes of a row. batch <= 8,
//   in % 32 == 0. Weight = code * scale (llama.cpp's Q8_0 rule, x2-facts D1, with codes and
//   scales in two arrays).
// The lab's cmake file (../shared-g12/qpieces.cmake) builds this file only when lab X2.1's
// solution is absent or has another signature; the driver prints which one it used.
//
// Not in the book, and ours: lab 22.1's gather shape (one warp per output, lanes over the
// row), each lane taking whole 32-code blocks, one float scale multiply per block. The codes
// of a block are 32 contiguous bytes at an offset that is a multiple of 32 (in % 32 == 0),
// so they are read as eight 4-byte words.
#include <cstdint>
#include <cstdio>

#include "../shared-g12/qformats.h"

#define G12_Q8_MAX_BATCH 8
#define G12_Q8_WARPS_PER_BLOCK 4

#if defined(__HIPCC__)
typedef unsigned long long g12_q8_lane_mask_t;
#else
typedef unsigned int g12_q8_lane_mask_t;
#endif

static __device__ float g12Q8HalfToFloat(uint16_t h) {
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

static __device__ float g12Q8WarpSum(float v) {
    const g12_q8_lane_mask_t full = ~(g12_q8_lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) v += __shfl_down_sync(full, v, offset);
    return v;
}

static int g12Q8DeviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

__global__ void gemvQ8_0G12Kernel(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in,
                                  int out) {
    const int lane = threadIdx.x % warpSize;
    const int j = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (j >= out) return;
    const int nb = in / QK8_0;
    float sum[G12_Q8_MAX_BATCH];
    for (int b = 0; b < G12_Q8_MAX_BATCH; ++b) sum[b] = 0.0f;
    for (int kb = lane; kb < nb; kb += warpSize) {
        const float d = g12Q8HalfToFloat(Wd[(size_t)j * nb + kb]);
        const int* q32 = (const int*)(Wq + (size_t)j * in + (size_t)kb * QK8_0);   // 4-byte aligned, see above
        float acc[G12_Q8_MAX_BATCH];
        for (int b = 0; b < G12_Q8_MAX_BATCH; ++b) acc[b] = 0.0f;
        for (int i = 0; i < 8; ++i) {
            const int w = q32[i];
            const float c0 = (float)(int8_t)(w & 0xFF), c1 = (float)(int8_t)((w >> 8) & 0xFF);
            const float c2 = (float)(int8_t)((w >> 16) & 0xFF), c3 = (float)(int8_t)((w >> 24) & 0xFF);
            for (int b = 0; b < G12_Q8_MAX_BATCH; ++b) {
                if (b < batch) {
                    const float* x = X + (size_t)b * in + (size_t)kb * QK8_0 + 4 * i;
                    acc[b] += c0 * x[0] + c1 * x[1] + c2 * x[2] + c3 * x[3];
                }
            }
        }
        for (int b = 0; b < G12_Q8_MAX_BATCH; ++b) sum[b] += d * acc[b];
    }
    for (int b = 0; b < G12_Q8_MAX_BATCH; ++b) {
        const float t = (b < batch) ? g12Q8WarpSum(sum[b]) : 0.0f;
        if (lane == 0 && b < batch) Y[(size_t)b * out + j] = t;
    }
}

void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0) return;
    if (batch > G12_Q8_MAX_BATCH || in % QK8_0 != 0) {
        std::fprintf(stderr, "launchGemvQ8_0: need batch <= %d and in %% 32 == 0 (got %d, %d)\n", G12_Q8_MAX_BATCH,
                     batch, in);
        return;
    }
    const int width = g12Q8DeviceWarpSize();
    if (width <= 0) {
        std::fprintf(stderr, "launchGemvQ8_0: could not read the device's warp width\n");
        return;
    }
    const int blocks = (out + G12_Q8_WARPS_PER_BLOCK - 1) / G12_Q8_WARPS_PER_BLOCK;
    gemvQ8_0G12Kernel<<<blocks, G12_Q8_WARPS_PER_BLOCK * width>>>(X, Wq, Wd, Y, batch, in, out);
}
