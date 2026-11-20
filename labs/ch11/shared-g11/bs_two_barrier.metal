// Provided for lab 11.1, not edited: Fig. 11.3 (p. 256) in Metal Shading Language.
// Kogge-Stone in one threadgroup buffer, two barriers per iteration. See
// bs_two_barrier.cu for the contract and where it differs from the page.
#include <metal_stdlib>
using namespace metal;

#define BS_MAX_BLOCK_DIM 1024

kernel void ks_two_barrier(device const float* input  [[buffer(0)]],
                           device float*       output [[buffer(1)]],
                           constant uint&      N      [[buffer(2)]],
                           uint t    [[thread_position_in_threadgroup]],
                           uint b    [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    threadgroup float buffer_s[BS_MAX_BLOCK_DIM];
    uint i = b * bdim + t;
    buffer_s[t] = i < N ? input[i] : 0.0f;
    for (uint stride = 1; stride < bdim; stride *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);   // last iteration's writes are done
        float temp = 0.0f;
        if (t >= stride) {
            temp = buffer_s[t] + buffer_s[t - stride];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // this iteration's reads are done
        if (t >= stride) {
            buffer_s[t] = temp;
        }
    }
    if (i < N) {
        output[i] = buffer_s[t];
    }
}
