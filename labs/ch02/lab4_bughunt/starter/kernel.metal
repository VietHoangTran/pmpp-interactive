// Lab 2.4 starter (Metal Shading Language). Not in the book.
//
// This kernel is dispatched in whole threadgroups (dispatchThreadgroups), the
// way llama.cpp's Metal backend dispatches its kernels. That makes it look
// like the CUDA kernel: it works out its own index from the same three values.
//
// CUDA          Metal
// blockIdx      [[threadgroup_position_in_grid]]
// threadIdx     [[thread_position_in_threadgroup]]
// blockDim      [[threads_per_threadgroup]]
//
// kernel.metal and launch.mm together have THREE independent defects (the
// CUDA version has a fourth that Metal has no counterpart for). They compile
// and they run. Find and fix all three.
#include <metal_stdlib>
using namespace metal;

kernel void scale_prefix(device float*   x   [[buffer(0)]],
                         constant float& s   [[buffer(1)]],
                         constant uint&  len [[buffer(2)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 local [[thread_position_in_threadgroup]],
                         uint3 groupSize [[threads_per_threadgroup]]) {
    uint i = group.x * groupSize.x + local.x;
    x[i] = s * x[i];
}
