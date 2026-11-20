// Provided for lab 11.6 (do not edit): the Metal counterpart of topp_copy.cu. Element i:
// read in[i], write outF[i] = in[i] and outB[i] = 1, the 9 bytes per element a top-p mask
// moves at least. One thread per element, every access coalesced, no arithmetic.
#include <metal_stdlib>
using namespace metal;

kernel void top_p_copy(device const float* in   [[buffer(0)]],
                       device float*       outF [[buffer(1)]],
                       device uchar*       outB [[buffer(2)]],
                       constant uint&      n    [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) {
        outF[i] = in[i];
        outB[i] = 1;
    }
}
