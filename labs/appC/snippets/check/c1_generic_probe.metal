// NOT valid MSL by its specification: rowMax's pointer names no address space, which
// section 4 of the specification makes a compile error (notes/today/appC-facts.md, C1).
// It exists only so that labs/appC/snippets/check.py can test the behaviour recorded
// under C1 as observed, not documented: at its default language version the Metal
// compiler on the build machine accepts it, and with -std=metal4.0 it does not.
// No lesson shows this file as code to copy. Write the overloads instead.
#include <metal_stdlib>
using namespace metal;

#define TILE 256

float rowMax(const float* p, uint n) {
    float m = p[0];
    for (uint i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One threadgroup (1 <= n <= TILE). out[0] from device memory, out[1] from a threadgroup
// copy holding x + 1, out[2] from a thread array holding x + 2.
kernel void generic_probe(device const float* x [[buffer(0)]],
                          device float* out     [[buffer(1)]],
                          constant uint& n      [[buffer(2)]],
                          uint t    [[thread_position_in_threadgroup]],
                          uint size [[threads_per_threadgroup]]) {
    threadgroup float tile[TILE];
    for (uint i = t; i < n; i += size) tile[i] = x[i] + 1.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        float mine[TILE];
        for (uint i = 0; i < n; ++i) mine[i] = x[i] + 2.0f;
        out[0] = rowMax(x, n);
        out[1] = rowMax(tile, n);
        out[2] = rowMax(mine, n);
    }
}
