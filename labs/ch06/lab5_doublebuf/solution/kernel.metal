// Lab 6.5 reference solution (Metal Shading Language).
// Double-buffered tiled matrix multiplication: one barrier per phase.
// Book: section 6.7, pp. 146-147, applied to lab 5.2's kernel. See kernel.cu for
// where this differs from the page. The driver defines TILE_WIDTH when it
// compiles this file (16 and 32 where the device takes them).
//
// Two pairs of TILE_WIDTH x TILE_WIDTH float tiles: 16 KB at TILE_WIDTH = 32. The
// driver checks the pipeline's threadgroup memory against the device's
// maxThreadgroupMemoryLength instead of assuming a limit.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

kernel void double_buffered_matmul(device const float* M [[buffer(0)]],
                                   device const float* N [[buffer(1)]],
                                   device float*       P [[buffer(2)]],
                                   constant uint&  Width [[buffer(3)]],
                                   uint3 tid [[thread_position_in_threadgroup]],
                                   uint3 bid [[threadgroup_position_in_grid]]) {
    // Two tiles per operand: phase ph loads into pair ph & 1.
    threadgroup float Mds[2][TILE_WIDTH][TILE_WIDTH];
    threadgroup float Nds[2][TILE_WIDTH][TILE_WIDTH];

    uint tx = tid.x, ty = tid.y;
    uint Row = bid.y * TILE_WIDTH + ty;
    uint Col = bid.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (uint ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        uint buf = ph & 1;
        Mds[buf][ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[buf][ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // read-after-write: the one barrier per phase

        for (uint k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[buf][ty][k] * Nds[buf][k][tx];
        }
        // No barrier here: the next phase writes the other pair, and the phase
        // after that cannot start writing this one until every thread has passed
        // the next phase's barrier, which is after this dot product.
    }
    P[Row * Width + Col] = Pvalue;
}
