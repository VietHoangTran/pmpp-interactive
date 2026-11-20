// Lab 5.2 reference solution (Metal Shading Language).
// The harness defines TILE_WIDTH when it compiles this file.
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx                    [[thread_position_in_threadgroup]]
// blockIdx                     [[threadgroup_position_in_grid]]
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

kernel void tiled_matmul(device const float* M [[buffer(0)]],
                         device const float* N [[buffer(1)]],
                         device float*       P [[buffer(2)]],
                         constant uint&  Width [[buffer(3)]],
                         uint3 tid [[thread_position_in_threadgroup]],
                         uint3 bid [[threadgroup_position_in_grid]]) {
    threadgroup float Mds[TILE_WIDTH][TILE_WIDTH];
    threadgroup float Nds[TILE_WIDTH][TILE_WIDTH];

    uint tx = tid.x, ty = tid.y;
    uint Row = bid.y * TILE_WIDTH + ty;
    uint Col = bid.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (uint ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    P[Row * Width + Col] = Pvalue;
}
