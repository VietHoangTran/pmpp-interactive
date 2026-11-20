// Provided for labs 11.1 and 11.2, not edited: a plain copy, the bandwidth baseline.
// See bs_copy.cu.
#include <metal_stdlib>
using namespace metal;

kernel void copy_kernel(device const float* input  [[buffer(0)]],
                        device float*       output [[buffer(1)]],
                        constant uint&      N      [[buffer(2)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < N) {
        output[i] = input[i];
    }
}
