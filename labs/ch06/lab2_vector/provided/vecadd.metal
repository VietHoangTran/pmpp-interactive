// Provided baseline for lab 6.2 on Metal: one element per thread. Same algorithm as
// provided/vecadd.cu. Do not edit.
#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* x [[buffer(0)]],
                    device const float* y [[buffer(1)]],
                    device float*       z [[buffer(2)]],
                    constant int&       n [[buffer(3)]],
                    uint gid [[thread_position_in_grid]]) {
    int i = int(gid);   // signed, like your kernels (see the note at the top of kernel.metal)
    if (i < n) z[i] = x[i] + y[i];
}
