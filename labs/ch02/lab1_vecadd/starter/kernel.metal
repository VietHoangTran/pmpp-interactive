// Lab 2.1 starter (Metal Shading Language). Not in the book.
//
// CUDA                                   Metal
// __global__ void                        kernel void
// float* (a raw device pointer)          device float* [[buffer(index)]]
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    // TODO: one pair-wise addition. The kernel has no n argument: decide in
    //       launch.mm how many threads run, then decide whether a guard is
    //       needed here.
}
