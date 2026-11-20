// Lab 7.1 reference solution (Metal Shading Language).
// The same three kernels as kernel.cu. Metal has no counterpart of __constant__ plus
// cudaMemcpyToSymbol (a program-scope `constant` array must be initialised in the
// source), so the three ways to hold the filter are:
//
//   CUDA                                   Metal
//   const float* F (global memory)         device const float* F    (a buffer)
//   __constant__ F_c + cudaMemcpyToSymbol  constant float* F        (a buffer in the
//                                                                    constant address space)
//   const Filter f, passed by value        constant Filter& f       (filled with setBytes)
//
// Position attributes are unsigned only, so the thread's position is converted to
// int before r is subtracted from it: a ghost cell has a negative index.
#include <metal_stdlib>
using namespace metal;

#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// @part basic
kernel void conv_basic(device const float* N [[buffer(0)]],
                       device const float* F [[buffer(1)]],
                       device float*       P [[buffer(2)]],
                       constant int&       r [[buffer(3)]],
                       constant int&   width [[buffer(4)]],
                       constant int&  height [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int outCol = int(gid.x);
    int outRow = int(gid.y);
    if (outRow >= height || outCol >= width) return;   // the dispatch rounds up to whole threadgroups

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

// @part const
// The only change from conv_basic is the address space of F: `constant` is Metal's
// read-only address space for data that many threads reuse. Apple documents no
// constant cache behind it.
kernel void conv_const(device const float* N [[buffer(0)]],
                       constant float*     F [[buffer(1)]],
                       device float*       P [[buffer(2)]],
                       constant int&       r [[buffer(3)]],
                       constant int&   width [[buffer(4)]],
                       constant int&  height [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    int outCol = int(gid.x);
    int outRow = int(gid.y);
    if (outRow >= height || outCol >= width) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

// @part byvalue
// A fixed-size struct taken by reference in the constant address space: the shape
// Apple's 2016 shader-optimisation talk recommends for "constant buffer preloading",
// because the reads are statically bounded. Whether a given Mac preloads it is not
// documented. It must match struct Filter in launch.mm byte for byte (904 bytes).
struct Filter {
    int r;
    float w[MAX_FILTER_DIM * MAX_FILTER_DIM];
};

kernel void conv_by_value(device const float* N [[buffer(0)]],
                          constant Filter&    f [[buffer(1)]],
                          device float*       P [[buffer(2)]],
                          constant int&   width [[buffer(3)]],
                          constant int&  height [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    int outCol = int(gid.x);
    int outRow = int(gid.y);
    if (outRow >= height || outCol >= width) return;

    const int r = f.r;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += f.w[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}
