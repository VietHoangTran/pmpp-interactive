// Lab 6.5 starter (Metal Shading Language).
// Book: section 6.7, pp. 146-147, applied to lab 5.2's kernel.
//
// This is lab 5.2's kernel with its second threadgroup_barrier deleted. It
// compiles, and it is wrong, although a GPU may hide that on many runs. The race
// backend runs kernel.cu, not this file: fix kernel.cu first, then make the same
// change here.
//
// Your task: make it correct with ONE threadgroup_barrier per phase. Do not put
// the second barrier back.
//
// The driver defines TILE_WIDTH when it compiles this file (16, and 32 where the
// device takes it), so do not hard-code 16 anywhere.
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
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
    // TODO 1: the same change to the tile declarations as in kernel.cu.
    threadgroup float Mds[TILE_WIDTH][TILE_WIDTH];
    threadgroup float Nds[TILE_WIDTH][TILE_WIDTH];

    uint tx = tid.x, ty = tid.y;
    uint Row = bid.y * TILE_WIDTH + ty;
    uint Col = bid.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (uint ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // TODO 2: decide where this phase's tiles go, and load them there.
        Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the one barrier per phase that you keep

        // TODO 3: read the tiles this phase loaded.
        for (uint k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        // Lab 5.2 had a second threadgroup_barrier here. It stays deleted.
    }
    P[Row * Width + Col] = Pvalue;
}
