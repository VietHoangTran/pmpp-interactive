// Lab 17.1 starter (Metal Shading Language).
// Same contract as kernel.cu; read its header first.
//
// CUDA                                         Metal
// blockIdx.x * blockDim.x + threadIdx.x        [[thread_position_in_grid]]
// a grid of whole blocks                       dispatchThreads in launch.mm makes exactly
//                                              as many threads as you ask for
// (the book's struct of device pointers)       one buffer binding per array
#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr(device const int*   rowPtrs [[buffer(0)]],
                     device const int*   colIdx  [[buffer(1)]],
                     device const float* values  [[buffer(2)]],
                     device const float* x       [[buffer(3)]],
                     device float*       y       [[buffer(4)]],
                     constant uint&      numRows [[buffer(5)]],
                     uint row [[thread_position_in_grid]]) {
    // TODO: TODOs 1 to 3 of kernel.cu, in Metal's vocabulary.
}

kernel void spmv_ell(device const int*   colIdx    [[buffer(0)]],
                     device const float* values    [[buffer(1)]],
                     device const int*   nnzPerRow [[buffer(2)]],
                     device const float* x         [[buffer(3)]],
                     device float*       y         [[buffer(4)]],
                     constant uint&      numRows   [[buffer(5)]],
                     constant uint&      width     [[buffer(6)]],
                     constant uint&      stride    [[buffer(7)]],
                     uint row [[thread_position_in_grid]]) {
    // TODO: TODOs 5 to 7 of kernel.cu.
}
