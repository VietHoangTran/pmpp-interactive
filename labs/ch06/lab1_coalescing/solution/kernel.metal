// Lab 6.1 reference solution (Metal Shading Language).
// The same three access patterns as kernel.cu. Indices are int, not uint:
// Apple advises signed integer types for array indexing in MSL.
//
// Apple documents no coalescing rule, and says threads are divided into
// SIMD-groups "in an implementation-defined fashion". So on Metal, whether
// neighbouring x threads behave like a warp reading neighbouring addresses is
// something this lab measures, not something it can predict.
#include <metal_stdlib>
using namespace metal;

struct StridedArgs { int n; int stride; };      // must match launch.mm
struct TransposeDims { int rows; int cols; };   // must match launch.mm

// ---------- 1. strided read: out[i] = in[i * stride] ----------
kernel void strided_read(device const float* in  [[buffer(0)]],
                         device float*       out [[buffer(1)]],
                         constant StridedArgs& a [[buffer(2)]],
                         uint3 gid [[thread_position_in_grid]]) {
    int i = int(gid.x);
    if (i < a.n) {
        out[i] = in[i * a.stride];
    }
}

// ---------- 2. transpose, rows read: x = column of in, y = row of in ----------
kernel void transpose_read_rows(device const float* in  [[buffer(0)]],
                                device float*       out [[buffer(1)]],
                                constant TransposeDims& d [[buffer(2)]],
                                uint3 gid [[thread_position_in_grid]]) {
    int col = int(gid.x);
    int row = int(gid.y);
    if (row < d.rows && col < d.cols) {
        out[col * d.rows + row] = in[row * d.cols + col];
    }
}

// ---------- 3. transpose, rows written: x = column of out (row of in) ----------
kernel void transpose_write_rows(device const float* in  [[buffer(0)]],
                                 device float*       out [[buffer(1)]],
                                 constant TransposeDims& d [[buffer(2)]],
                                 uint3 gid [[thread_position_in_grid]]) {
    int row = int(gid.x);   // row of in = column of out
    int col = int(gid.y);   // column of in = row of out
    if (row < d.rows && col < d.cols) {
        out[col * d.rows + row] = in[row * d.cols + col];
    }
}
