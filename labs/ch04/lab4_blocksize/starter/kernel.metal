// Lab 4.4 starter (Metal Shading Language).
// The embedding-bag gather of kernel.cu:
//   out[i] = sum over j < k of table[idx[i*k + j]]      for i < n
//
// CUDA                                  Metal
// blockIdx.x*blockDim.x + threadIdx.x   [[thread_position_in_grid]], a uint for a 1D grid
// n and k passed by value               a constant struct, bound with setBytes in launch.mm
// threads per block in <<< >>>          threadsPerThreadgroup: in launch.mm
#include <metal_stdlib>
using namespace metal;

struct GatherDims { uint n; uint k; };

kernel void gather_sum(device const float* table [[buffer(0)]],
                       device const int*   idx   [[buffer(1)]],
                       device float*       out   [[buffer(2)]],
                       constant GatherDims& d    [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    // TODO 1: one thread per output. Whether a guard is needed depends on how
    //         launch.mm dispatches.
}
