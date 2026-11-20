// Provided baselines for lab 10.1 on Metal, not edited: the same two kernels as
// shared/inplace_sum.cu (book Figs. 10.5 and 10.8), one row of 2*block floats per
// threadgroup, written in place, result of row b in output[b].
//
// These two kernels exchange partial sums through DEVICE memory, not threadgroup
// memory, so their barrier must order device-memory accesses:
// threadgroup_barrier(mem_flags::mem_device). With mem_threadgroup the barrier would
// order only threadgroup memory, which these kernels do not use
// (notes/today/ch05-facts.md C2).
#include <metal_stdlib>
using namespace metal;

kernel void simple_sum(device float* input  [[buffer(0)]],
                       device float* output [[buffer(1)]],
                       uint t    [[thread_position_in_threadgroup]],
                       uint b    [[threadgroup_position_in_grid]],
                       uint bdim [[threads_per_threadgroup]]) {
    device float* row = input + 2 * bdim * b;
    uint i = 2 * t;
    for (uint stride = 1; stride <= bdim; stride *= 2) {
        if (t % stride == 0) {
            row[i] += row[i + stride];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (t == 0) {
        output[b] = row[0];
    }
}

kernel void convergent_sum(device float* input  [[buffer(0)]],
                           device float* output [[buffer(1)]],
                           uint t    [[thread_position_in_threadgroup]],
                           uint b    [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    device float* row = input + 2 * bdim * b;
    uint i = t;
    for (uint stride = bdim; stride >= 1; stride /= 2) {
        if (t < stride) {
            row[i] += row[i + stride];
        }
        threadgroup_barrier(mem_flags::mem_device);
    }
    if (t == 0) {
        output[b] = row[0];
    }
}
