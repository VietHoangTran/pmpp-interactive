#include <metal_stdlib>
using namespace metal;

#define TILE 256

// MSL has no generic address space: every pointer names its memory, and the
// address space is part of the overload. One definition per memory you call it on.
float rowMax(device const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

float rowMax(threadgroup const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One threadgroup per row of x (row-major, n columns, 1 <= n <= TILE). Thread 0
// takes the row's maximum twice: from device memory, and from the threadgroup copy.
kernel void row_max_twice(device const float* x            [[buffer(0)]],
                          device float* fromDevice         [[buffer(1)]],
                          device float* fromThreadgroup    [[buffer(2)]],
                          constant uint& n                 [[buffer(3)]],
                          uint row  [[threadgroup_position_in_grid]],
                          uint t    [[thread_position_in_threadgroup]],
                          uint size [[threads_per_threadgroup]]) {
    threadgroup float tile[TILE];
    device const float* r = x + row * n;
    for (uint i = t; i < n; i += size) tile[i] = r[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        fromDevice[row] = rowMax(r, n);            // the device overload
        fromThreadgroup[row] = rowMax(tile, n);    // the threadgroup overload
    }
}
