// Lab 2.4 reference solution (Metal Shading Language): the kernel's defect, fixed.
// Not in the book.
//
// This kernel is dispatched in whole threadgroups (dispatchThreadgroups), the
// way llama.cpp's Metal backend dispatches its kernels. That makes it look
// like the CUDA kernel: it works out its own index, and it needs the guard.
//
// CUDA          Metal
// blockIdx      [[threadgroup_position_in_grid]]
// threadIdx     [[thread_position_in_threadgroup]]
// blockDim      [[threads_per_threadgroup]]
#include <metal_stdlib>
using namespace metal;

kernel void scale_prefix(device float*   x   [[buffer(0)]],
                         constant float& s   [[buffer(1)]],
                         constant uint&  len [[buffer(2)]],
                         uint3 group [[threadgroup_position_in_grid]],
                         uint3 local [[thread_position_in_threadgroup]],
                         uint3 groupSize [[threads_per_threadgroup]]) {
    uint i = group.x * groupSize.x + local.x;
    // FIX 1: the guard. The grid is whole threadgroups, so it has up to
    // groupSize.x - 1 threads more than len. The starter let them run: they
    // scaled the padding, and when len was n they wrote past the end of the
    // buffer.
    if (i < len) {
        x[i] = s * x[i];
    }
}
