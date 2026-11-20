// Lab 6.4 starter (Metal Shading Language).
// Y[batch x out] = X[batch x in] * W^T, W stored (out, in) row-major.
// Same task as kernel.cu: the X tile is done; the W tile (declaration, corner-
// turned load, and its read in the product loop) is yours. The harness defines
// TILE_WIDTH when it compiles this file (16 and 32), so do not hard-code 16.
//
// Apple documents no coalescing rule and no threadgroup-memory banks: on Metal,
// what corner turning and padding are worth is something you measure.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

struct LinearDims { int batch; int in; int out; };   // must match launch.mm

kernel void linear(device const float* X [[buffer(0)]],
                   device const float* W [[buffer(1)]],
                   device float*       Y [[buffer(2)]],
                   constant LinearDims& d [[buffer(3)]],
                   uint3 tid [[thread_position_in_threadgroup]],
                   uint3 bid [[threadgroup_position_in_grid]]) {
    threadgroup float Xs[TILE_WIDTH][TILE_WIDTH];
    // TODO 1: declare the threadgroup tile for W.

    int tx = int(tid.x), ty = int(tid.y);
    int Row = int(bid.y) * TILE_WIDTH + ty;
    int Col = int(bid.x) * TILE_WIDTH + tx;

    float Yvalue = 0;
    for (int ph = 0; ph < (d.in + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int xk = ph * TILE_WIDTH + tx;
        Xs[ty][tx] = (Row < d.batch && xk < d.in) ? X[Row * d.in + xk] : 0.0f;

        // TODO 2: load this phase's W tile, one element per thread, corner-turned.
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < TILE_WIDTH; ++i) {
            // TODO 3: add X(Row, ph*T + i) * W(Col, ph*T + i), both read from threadgroup memory.
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (Row < d.batch && Col < d.out) {
        Y[Row * d.out + Col] = Yvalue;
    }
}
