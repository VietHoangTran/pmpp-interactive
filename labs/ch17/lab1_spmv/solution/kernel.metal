// Lab 17.1 reference solution (Metal Shading Language).
// Same contract and design as kernel.cu; read its header first.
//
// CUDA                                         Metal
// blockIdx.x * blockDim.x + threadIdx.x        [[thread_position_in_grid]]
// a grid of whole blocks, so row < numRows     dispatchThreads makes exactly numRows threads
//   decides who works                          (notes/today/ch05-facts.md C4, C5)
// the book's struct of device pointers         one buffer binding per array (ch17-facts C2)
#include <metal_stdlib>
using namespace metal;

kernel void spmv_csr(device const int*   rowPtrs [[buffer(0)]],
                     device const int*   colIdx  [[buffer(1)]],
                     device const float* values  [[buffer(2)]],
                     device const float* x       [[buffer(3)]],
                     device float*       y       [[buffer(4)]],
                     constant uint&      numRows [[buffer(5)]],
                     uint row [[thread_position_in_grid]]) {
    // Never false with dispatchThreads; it keeps the kernel right if someone dispatches
    // whole threadgroups instead.
    if (row < numRows) {
        float sum = 0.0f;
        for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; ++i) {
            int col = colIdx[i];
            float value = values[i];
            sum += x[col] * value;
        }
        y[row] += sum;
    }
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
    // `width` is not needed when the loop stops at nnzPerRow[row].
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < nnzPerRow[row]; ++t) {
            uint i = t * stride + row;
            int col = colIdx[i];
            float value = values[i];
            sum += x[col] * value;
        }
        y[row] += sum;
    }
}
