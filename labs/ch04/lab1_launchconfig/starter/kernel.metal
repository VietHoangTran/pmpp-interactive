// Lab 4.1 starter (Metal Shading Language).
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

// ---------- your part: the probe ----------
// Every thread writes down which SIMD-group it is in, which lane of that group
// it has, and how wide the group is: three uints per thread, in that order, at
// rec[3 * p], where p is the thread's row-major position in a gridWidth-wide grid.
kernel void simd_probe(device uint* rec [[buffer(0)]],
                       constant uint& gridWidth [[buffer(1)]],
                       uint2 gid [[thread_position_in_grid]]
                       // TODO 1: three more parameters, one per attribute in the
                       //         table above. They must all have the same type.
                       ) {
    // TODO 2: store the three values.
}
