// Lab 6.3 starter (Metal Shading Language).
// The harness defines TILE_WIDTH when it compiles this file (16, and 32 where the
// pipeline accepts 1024 threads per threadgroup), so do not hard-code 16.
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx                    [[thread_position_in_threadgroup]]
// blockIdx                     [[threadgroup_position_in_grid]]
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

kernel void tiled_transpose(device const float* in    [[buffer(0)]],
                            device float*       out   [[buffer(1)]],
                            constant int&       rows  [[buffer(2)]],
                            constant int&       cols  [[buffer(3)]],
                            constant int&       pitch [[buffer(4)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 bid [[threadgroup_position_in_grid]]) {
    // Big enough for the larger pitch. Index it as tile[row * pitch + col].
    threadgroup float tile[TILE_WIDTH * (TILE_WIDTH + 1)];

    // TODO: the same three steps as tiledTransposeKernel in kernel.cu, in Metal's
    //       vocabulary. Every thread must reach the barrier.
}
