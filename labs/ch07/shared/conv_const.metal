// Lab 7.2, provided: the baseline the tiled kernels are measured against. Do not edit.
// (Metal Shading Language.) The Metal counterpart of conv_const.cu: book Fig. 7.9,
// one thread per output, the filter in the constant address space, every input
// element read from device memory. The driver compiles one pipeline per radius,
// passing FILTER_RADIUS as a preprocessor macro, so the filter loops have fixed
// trip counts, as in your kernels. Threadgroups are always 16 x 16.
#include <metal_stdlib>
using namespace metal;

#ifndef FILTER_RADIUS
#define FILTER_RADIUS 2
#endif
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

kernel void conv_const_base(device const float* N [[buffer(0)]],
                            constant float*     F [[buffer(1)]],
                            device float*       P [[buffer(2)]],
                            constant int&   width [[buffer(3)]],
                            constant int&  height [[buffer(4)]],
                            uint2 gid [[thread_position_in_grid]]) {
    int outCol = int(gid.x);
    int outRow = int(gid.y);
    if (outRow >= height || outCol >= width) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
        for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * FILTER_DIM + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}
