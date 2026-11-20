// Lab 19.1 reference solution (Metal Shading Language).
// The direct kernel (Fig. 19.7 corrected) and the unroll kernel, as in kernel.cu.
// The GEMM is lab 6.4's `linear` kernel, which launch.mm binds through encodeLinearAt.
//
// CUDA                         Metal
// blockIdx                     [[threadgroup_position_in_grid]]
// threadIdx                    [[thread_position_in_threadgroup]]
// blockIdx * blockDim + threadIdx   [[thread_position_in_grid]]
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// The layer's sizes. W_grid is used by conv_direct only; unroll_input ignores M and W_grid.
struct ConvDims { int C; int H; int W; int M; int K; int W_grid; };   // must match launch.mm

// [lesson:direct]
// One thread per output pixel; threadgroup (m, tile, n), the tile index linearised
// row-major with W_grid tiles per row, rounded up, so the thread checks its pixel.
kernel void conv_direct(device const float* X [[buffer(0)]],
                        device const float* F [[buffer(1)]],
                        device float*       Y [[buffer(2)]],
                        constant ConvDims&  d [[buffer(3)]],
                        uint3 tid [[thread_position_in_threadgroup]],
                        uint3 bid [[threadgroup_position_in_grid]]) {
    int C = d.C, H = d.H, W = d.W, M = d.M, K = d.K;
    int H_out = H - K + 1, W_out = W - K + 1;
    int m = int(bid.x);
    int h = (int(bid.y) / d.W_grid) * TILE_WIDTH + int(tid.y);
    int w = (int(bid.y) % d.W_grid) * TILE_WIDTH + int(tid.x);
    int n = int(bid.z);
    if (h < H_out && w < W_out) {
        float acc = 0.0f;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += X[((n * C + c) * H + (h + p)) * W + (w + q)]
                         * F[((m * C + c) * K + p) * K + q];
                }
            }
        }
        Y[((n * M + m) * H_out + h) * W_out + w] = acc;
    }
}
// [/lesson:direct]

// [lesson:unroll]
// One thread per element of Bt = B^T (P rows of C*K*K), in Bt's row-major order.
// Xn is sample n's C x H x W block: launch.mm binds X at a byte offset.
kernel void unroll_input(device const float* Xn [[buffer(0)]],
                         device float*       Bt [[buffer(1)]],
                         constant ConvDims&  d  [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    int C = d.C, H = d.H, W = d.W, K = d.K;
    int W_out = W - K + 1;
    int P = (H - K + 1) * W_out;
    int CKK = C * K * K;
    if (int(i) < P * CKK) {
        int v = int(i) / CKK, u = int(i) % CKK;
        int c = u / (K * K), r = u % (K * K);
        int h = v / W_out + r / K;
        int w = v % W_out + r % K;
        Bt[i] = Xn[(c * H + h) * W + w];
    }
}
// [/lesson:unroll]
