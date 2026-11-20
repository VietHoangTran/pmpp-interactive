// Lab 7.2 reference solution (Metal Shading Language).
// Tiled 2D convolution, two ways. Book: section 7.5, Fig. 7.12 (p. 174), and
// section 7.6, Fig. 7.15 (p. 178).
//
// The driver compiles one pipeline per tile width and radius, passing TILE_WIDTH and
// FILTER_RADIUS as preprocessor macros. They size the threadgroup arrays, which a
// Metal function constant cannot do (notes/today/ch07-facts.md C3).
// The filter arrives as a buffer in the constant address space, Metal's counterpart
// of a __constant__ array that the host fills (C1).
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx                    [[thread_position_in_threadgroup]]   (unsigned only)
// blockIdx                     [[threadgroup_position_in_grid]]     (unsigned only)
//
// Where this file differs from the pages: as in kernel.cu. Positions are converted
// to int before anything is subtracted from them, and Fig. 7.15's shared-memory
// index gets the - FILTER_RADIUS its guard already uses.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#ifndef FILTER_RADIUS
#define FILTER_RADIUS 2
#endif
#define FILTER_DIM (2 * FILTER_RADIUS + 1)

// @part halo
// Part A: the threadgroup is the input tile.
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

kernel void conv_tiled_halo(device const float* N [[buffer(0)]],
                            constant float*     F [[buffer(1)]],
                            device float*       P [[buffer(2)]],
                            constant int&   width [[buffer(3)]],
                            constant int&  height [[buffer(4)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 bid [[threadgroup_position_in_grid]]) {
    int tx = int(tid.x), ty = int(tid.y);
    int col = int(bid.x) * OUT_TILE_DIM + tx - FILTER_RADIUS;
    int row = int(bid.y) * OUT_TILE_DIM + ty - FILTER_RADIUS;

    threadgroup float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[ty][tx] = N[row * width + col];
    } else {
        N_s[ty][tx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
                for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
                    Pvalue += F[fRow * FILTER_DIM + fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

// @part cached
// Part B: threadgroup memory holds only the tile; halo taps come from device memory.
#define TILE_DIM TILE_WIDTH

kernel void conv_tiled_cached(device const float* N [[buffer(0)]],
                              constant float*     F [[buffer(1)]],
                              device float*       P [[buffer(2)]],
                              constant int&   width [[buffer(3)]],
                              constant int&  height [[buffer(4)]],
                              uint2 tid [[thread_position_in_threadgroup]],
                              uint2 bid [[threadgroup_position_in_grid]]) {
    int tx = int(tid.x), ty = int(tid.y);
    int col = int(bid.x) * TILE_DIM + tx;
    int row = int(bid.y) * TILE_DIM + ty;

    threadgroup float N_s[TILE_DIM][TILE_DIM];
    if (row < height && col < width) {
        N_s[ty][tx] = N[row * width + col];
    } else {
        N_s[ty][tx] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
            for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
                if (tx - FILTER_RADIUS + fCol >= 0 &&
                    tx - FILTER_RADIUS + fCol < TILE_DIM &&
                    ty - FILTER_RADIUS + fRow >= 0 &&
                    ty - FILTER_RADIUS + fRow < TILE_DIM) {
                    Pvalue += F[fRow * FILTER_DIM + fCol] *
                              N_s[ty - FILTER_RADIUS + fRow][tx - FILTER_RADIUS + fCol];
                } else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {
                        Pvalue += F[fRow * FILTER_DIM + fCol] *
                                  N[(row - FILTER_RADIUS + fRow) * width + col - FILTER_RADIUS + fCol];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}
