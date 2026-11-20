// Lab 4.1 reference solution (Metal Shading Language).
//
// CUDA                                Metal
// warp                                SIMD-group
// warpSize in a kernel                [[threads_per_simdgroup]]
// linear thread index % warpSize      [[thread_index_in_simdgroup]]      (the lane)
// linear thread index / warpSize      [[simdgroup_index_in_threadgroup]] (which warp)
//
// The last two rows are arithmetic on CUDA, where the cut of a block into warps
// is documented. Metal's specification calls the cut of a threadgroup into
// SIMD-groups implementation-defined, so a Metal kernel asks instead of
// computing. The probe below is how you find out what your GPU does.
#include <metal_stdlib>
using namespace metal;

struct LinearDims { uint batch; uint in; uint out; };

// ---------- given: the kernel ----------
// Y[b][o] = sum over k of X[b][k] * W[k][o], one thread per output.
// There is no guard. launch.mm must use dispatchThreads, which starts exactly
// out x batch threads, so no thread lies outside Y.
kernel void linear_layer(device const float* X [[buffer(0)]],
                         device const float* W [[buffer(1)]],
                         device float*       Y [[buffer(2)]],
                         constant LinearDims& d [[buffer(3)]],
                         uint2 gid [[thread_position_in_grid]]) {
    uint b = gid.y;   // row of Y
    uint o = gid.x;   // column of Y
    float y = 0;
    for (uint k = 0; k < d.in; ++k) {
        y += X[b * d.in + k] * W[k * d.out + o];
    }
    Y[b * d.out + o] = y;
}

// ---------- the probe ----------
// Every thread writes down which SIMD-group it is in, which lane of that group
// it has, and how wide the group is. Three uints per thread, at the thread's
// position in a gridWidth-wide grid. The three attributes must be declared
// with the same type.
kernel void simd_probe(device uint* rec [[buffer(0)]],
                       constant uint& gridWidth [[buffer(1)]],
                       uint2 gid   [[thread_position_in_grid]],
                       uint  group [[simdgroup_index_in_threadgroup]],
                       uint  lane  [[thread_index_in_simdgroup]],
                       uint  width [[threads_per_simdgroup]]) {
    uint i = 3 * (gid.y * gridWidth + gid.x);
    rec[i]     = group;
    rec[i + 1] = lane;
    rec[i + 2] = width;
}
