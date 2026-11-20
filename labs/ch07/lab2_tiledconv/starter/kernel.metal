// Lab 7.2 starter (Metal Shading Language).
// Tiled 2D convolution, two ways. Book: section 7.5 (pp. 172-177) and section 7.6
// (pp. 178-179).
//
// Part A: write conv_tiled_halo (after Fig. 7.12, p. 174).
// Part B: repair conv_tiled_cached (after Fig. 7.15, p. 178). It compiles and has the
//         same two defects as the CUDA version in kernel.cu.
//
// The driver compiles one pipeline per tile width and radius, passing TILE_WIDTH (8,
// 16 and 32) and FILTER_RADIUS as preprocessor macros, so do not hard-code either.
// Macros can size threadgroup arrays; a Metal function constant cannot
// (notes/today/ch07-facts.md C3). The filter arrives as a buffer in the constant
// address space, Metal's counterpart of a __constant__ array the host fills (C1),
// indexed fRow * FILTER_DIM + fCol.
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx                    [[thread_position_in_threadgroup]]   (unsigned only)
// blockIdx                     [[threadgroup_position_in_grid]]     (unsigned only)
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
// Part A: the threadgroup is the input tile, one thread per input-tile element.
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM (IN_TILE_DIM - 2 * FILTER_RADIUS)

kernel void conv_tiled_halo(device const float* N [[buffer(0)]],
                            constant float*     F [[buffer(1)]],
                            device float*       P [[buffer(2)]],
                            constant int&   width [[buffer(3)]],
                            constant int&  height [[buffer(4)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 bid [[threadgroup_position_in_grid]]) {
    // TODO: the same four steps as convTiledHaloKernel in kernel.cu, in Metal's
    //       vocabulary. tid and bid are unsigned: convert them to int before you
    //       subtract anything from them.
}

// @part cached
// Part B: repair this kernel. Threadgroup memory holds only the tile itself; halo
// taps are read from device memory.
#define TILE_DIM TILE_WIDTH

kernel void conv_tiled_cached(device const float* N [[buffer(0)]],
                              constant float*     F [[buffer(1)]],
                              device float*       P [[buffer(2)]],
                              constant int&   width [[buffer(3)]],
                              constant int&  height [[buffer(4)]],
                              uint2 tid [[thread_position_in_threadgroup]],
                              uint2 bid [[threadgroup_position_in_grid]]) {
    int col = bid.x * TILE_DIM + tid.x;
    int row = bid.y * TILE_DIM + tid.y;

    // loading input tile
    threadgroup float N_s[TILE_DIM][TILE_DIM];
    if (row < height) {
        N_s[tid.y][tid.x] = N[row * width + col];
    } else {
        N_s[tid.y][tid.x] = 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // calculating output elements
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < FILTER_DIM; fRow++) {
            for (int fCol = 0; fCol < FILTER_DIM; fCol++) {
                if (tid.x - FILTER_RADIUS + fCol >= 0 &&
                    tid.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    tid.y - FILTER_RADIUS + fRow >= 0 &&
                    tid.y - FILTER_RADIUS + fRow < TILE_DIM) {
                    Pvalue += F[fRow * FILTER_DIM + fCol] * N_s[tid.y + fRow][tid.x + fCol];
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
