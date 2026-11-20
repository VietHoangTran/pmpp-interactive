// Lab 5.3 reference solution (Metal Shading Language).
// C[j x l] = A[j x k] * B[k x l], row-major, any sizes.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

struct GemmDims { uint j; uint k; uint l; };

kernel void gemm(device const float* A [[buffer(0)]],
                 device const float* B [[buffer(1)]],
                 device float*       C [[buffer(2)]],
                 constant GemmDims&  d [[buffer(3)]],
                 uint3 tid [[thread_position_in_threadgroup]],
                 uint3 bid [[threadgroup_position_in_grid]]) {
    threadgroup float As[TILE_WIDTH][TILE_WIDTH];
    threadgroup float Bs[TILE_WIDTH][TILE_WIDTH];

    uint tx = tid.x, ty = tid.y;
    uint Row = bid.y * TILE_WIDTH + ty;
    uint Col = bid.x * TILE_WIDTH + tx;

    float Cvalue = 0;
    for (uint ph = 0; ph < (d.k + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        uint aCol = ph * TILE_WIDTH + tx;
        uint bRow = ph * TILE_WIDTH + ty;
        As[ty][tx] = (Row < d.j && aCol < d.k) ? A[Row * d.k + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < d.k && Col < d.l) ? B[bRow * d.l + Col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (Row < d.j && Col < d.l) {
        C[Row * d.l + Col] = Cvalue;
    }
}
