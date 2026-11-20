// Lab 6.1 starter (Metal Shading Language).
// The same three access patterns as kernel.cu. Use int for indices, not uint:
// Apple advises signed integer types for array indexing in MSL.
//
// CUDA                                        Metal
// blockIdx.x * blockDim.x + threadIdx.x       [[thread_position_in_grid]].x
// blockIdx.y * blockDim.y + threadIdx.y       [[thread_position_in_grid]].y
#include <metal_stdlib>
using namespace metal;

struct StridedArgs { int n; int stride; };      // must match launch.mm
struct TransposeDims { int rows; int cols; };   // must match launch.mm

// 1. Strided read: out[i] = in[i * stride] for every i < n.
kernel void strided_read(device const float* in  [[buffer(0)]],
                         device float*       out [[buffer(1)]],
                         constant StridedArgs& a [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]]) {
    // TODO
}

// 2. Transpose, rows read: x is a column of `in`, y a row of `in`.
//    `in` is rows x cols, `out` is cols x rows, out(c, r) = in(r, c).
kernel void transpose_read_rows(device const float* in  [[buffer(0)]],
                                device float*       out [[buffer(1)]],
                                constant TransposeDims& d [[buffer(2)]],
                                uint3 gid [[thread_position_in_grid]]) {
    // TODO
}

// 3. Transpose, rows written: x is a column of `out`, y a row of `out`.
kernel void transpose_write_rows(device const float* in  [[buffer(0)]],
                                 device float*       out [[buffer(1)]],
                                 constant TransposeDims& d [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    // TODO
}
