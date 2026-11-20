// Lab 5.2 starter (Metal Shading Language).
// The harness defines TILE_WIDTH when it compiles this file (it tests 16 and 8,
// and benchmarks 4 to 32), so do not hard-code 16 anywhere.
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

kernel void tiled_matmul(device const float* M [[buffer(0)]],
                         device const float* N [[buffer(1)]],
                         device float*       P [[buffer(2)]],
                         constant uint&  Width [[buffer(3)]],
                         uint3 tid [[thread_position_in_threadgroup]],
                         uint3 bid [[threadgroup_position_in_grid]]) {
    // TODO: the same five steps as in kernel.cu, in Metal's vocabulary.
}
