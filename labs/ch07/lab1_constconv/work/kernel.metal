// Lab 7.1 starter (Metal Shading Language).
// The same three kernels as kernel.cu. Metal has no counterpart of __constant__ plus
// cudaMemcpyToSymbol (a program-scope `constant` array must be initialised in the
// source), so here the three ways to hold the filter are three kinds of kernel
// argument:
//
//   CUDA                                   Metal
//   const float* F (global memory)         a buffer in the device address space
//   __constant__ F_c + cudaMemcpyToSymbol  a buffer in the constant address space
//   const Filter f, passed by value        a Filter struct, filled with setBytes
//
// CUDA's blockIdx * blockDim + threadIdx is given directly by [[thread_position_in_grid]].
#include <metal_stdlib>
using namespace metal;

#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// @part basic
kernel void conv_basic(device const float* N [[buffer(0)]],
                       device const float* F [[buffer(1)]],
                       device float*       P [[buffer(2)]],
                       constant int&       r [[buffer(3)]],
                       constant int&   width [[buffer(4)]],
                       constant int&  height [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // TODO 1: the same steps as convBasicKernel in kernel.cu. gid is unsigned: a
    //         ghost cell's index is negative, so convert before you subtract r.
}

// @part const
kernel void conv_const(device const float* N [[buffer(0)]],
                       // TODO 2: the filter at buffer(1), declared in the address space
                       //         Metal provides for read-only data that many threads reuse.
                       device float*       P [[buffer(2)]],
                       constant int&       r [[buffer(3)]],
                       constant int&   width [[buffer(4)]],
                       constant int&  height [[buffer(5)]],
                       uint2 gid [[thread_position_in_grid]]) {
    // TODO 3: as conv_basic.
}

// @part byvalue
// Provided. It must match struct Filter in launch.mm byte for byte (904 bytes).
struct Filter {
    int r;
    float w[MAX_FILTER_DIM * MAX_FILTER_DIM];
};

kernel void conv_by_value(device const float* N [[buffer(0)]],
                          // TODO 4: the Filter at buffer(1). setBytes fills it, and a
                          //         kernel argument filled that way is a reference in
                          //         the constant address space.
                          device float*       P [[buffer(2)]],
                          constant int&   width [[buffer(3)]],
                          constant int&  height [[buffer(4)]],
                          uint2 gid [[thread_position_in_grid]]) {
    // TODO 5: as conv_basic, with the radius and the weights taken from the Filter.
}
