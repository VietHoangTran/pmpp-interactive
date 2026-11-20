// Provided baseline for Metal: lab 5.2's tiled matrix multiplication, two barriers
// per phase. Same algorithm as provided/tiled_two_barriers.cu. The driver defines
// TILE_WIDTH when it compiles this file.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

kernel void two_barrier_matmul(device const float* M [[buffer(0)]],
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
        threadgroup_barrier(mem_flags::mem_threadgroup);   // read-after-write

        for (uint k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // write-after-read
    }
    P[Row * Width + Col] = Pvalue;
}
