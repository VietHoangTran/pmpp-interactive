// Lab 5.3 starter (Metal Shading Language).
// C[j x l] = A[j x k] * B[k x l], row-major, ANY sizes.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

struct GemmDims { uint j; uint k; uint l; };   // must match the struct in launch.mm

kernel void gemm(device const float* A [[buffer(0)]],
                 device const float* B [[buffer(1)]],
                 device float*       C [[buffer(2)]],
                 constant GemmDims&  d [[buffer(3)]],
                 uint3 tid [[thread_position_in_threadgroup]],
                 uint3 bid [[threadgroup_position_in_grid]]) {
    // TODO
}
