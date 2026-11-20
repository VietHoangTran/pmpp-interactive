// Lab 2.1 reference solution (Metal Shading Language). Not in the book.
//
// CUDA                                   Metal
// __global__ void                        kernel void
// float* (a raw device pointer)          device float* [[buffer(index)]]
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// if (i < n)                             not needed when the host dispatches exactly n threads
#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    // No index formula: Metal hands each thread its position in the grid.
    // No guard: launch.mm dispatches n threads, so every i is below n.
    C[i] = A[i] + B[i];
}
