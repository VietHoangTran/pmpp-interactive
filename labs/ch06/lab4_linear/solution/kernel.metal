// Lab 6.4 reference solution (Metal Shading Language).
// Y[batch x out] = X[batch x in] * W^T, W stored (out, in) row-major, corner-turned
// and padded as in kernel.cu.
//
// Apple documents no coalescing rule and no threadgroup-memory banks, so on Metal
// both changes are things you measure, not things the documentation predicts.
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
    threadgroup float Ws[TILE_WIDTH][TILE_WIDTH + 1];   // Ws[i][j] = W(o = bx*T + j, k = ph*T + i)

    int tx = int(tid.x), ty = int(tid.y);
    int Row = int(bid.y) * TILE_WIDTH + ty;
    int Col = int(bid.x) * TILE_WIDTH + tx;

    float Yvalue = 0;
    for (int ph = 0; ph < (d.in + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int xk = ph * TILE_WIDTH + tx;
        Xs[ty][tx] = (Row < d.batch && xk < d.in) ? X[Row * d.in + xk] : 0.0f;

        int wo = int(bid.x) * TILE_WIDTH + ty;   // corner-turned: x walks along a row of W
        int wk = ph * TILE_WIDTH + tx;
        Ws[tx][ty] = (wo < d.out && wk < d.in) ? W[wo * d.in + wk] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Yvalue += Xs[ty][i] * Ws[i][tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (Row < d.batch && Col < d.out) {
        Y[Row * d.out + Col] = Yvalue;
    }
}
