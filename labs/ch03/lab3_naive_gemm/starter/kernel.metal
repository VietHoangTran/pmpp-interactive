// Lab 3.3 starter (Metal Shading Language).
// Naive GEMM, one thread per element of C:
//   C[j x l] = A[j x k] * B[k x l], row-major float, any sizes.
//
// CUDA                                  Metal
// blockIdx*blockDim + threadIdx         [[thread_position_in_grid]]: .x is the column, .y the row
// int arguments passed by value         a constant struct, bound with setBytes in launch.mm
// kernel<<<dimGrid, dimBlock>>>(...)    dispatchThreadgroups:threadsPerThreadgroup: in launch.mm
#include <metal_stdlib>
using namespace metal;

struct GemmDims { uint j; uint k; uint l; };

kernel void naive_gemm(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float*       C [[buffer(2)]],
                       constant GemmDims&  d [[buffer(3)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // TODO: the same four steps as in kernel.cu. Metal hands you the thread's
    //       position in the whole grid, so step 1 needs no arithmetic. Whether
    //       step 2 is needed depends on how launch.mm dispatches.
}
