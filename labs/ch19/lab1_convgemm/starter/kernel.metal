// Lab 19.1 starter (Metal Shading Language).
// The direct kernel and the unroll kernel, as in kernel.cu. The GEMM is lab 6.4's
// `linear` kernel, which launch.mm binds through the provided encodeLinearAt.
// The harness defines TILE_WIDTH (16) when it compiles this file.
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

// The layer's sizes. W_grid is for conv_direct; unroll_input can ignore M and W_grid.
struct ConvDims { int C; int H; int W; int M; int K; int W_grid; };   // must match launch.mm

// One thread per output pixel; threadgroup (m, tile, n), as kernel.cu's convDirectKernel.
kernel void conv_direct(device const float* X [[buffer(0)]],
                        device const float* F [[buffer(1)]],
                        device float*       Y [[buffer(2)]],
                        constant ConvDims&  d [[buffer(3)]],
                        uint3 tid [[thread_position_in_threadgroup]],
                        uint3 bid [[threadgroup_position_in_grid]]) {
    // TODO: TODOs 1 to 3 of kernel.cu, in Metal's vocabulary.
}

// Sample n's B^T into Bt (P rows of C*K*K). Xn is sample n's C x H x W block: your
// encodeUnroll decides how buffer 0 gets there.
kernel void unroll_input(device const float* Xn [[buffer(0)]],
                         device float*       Bt [[buffer(1)]],
                         constant ConvDims&  d  [[buffer(2)]],
                         uint i [[thread_position_in_grid]]) {
    // TODO: TODO 5 of kernel.cu. The dispatch rounds the thread count up.
}
