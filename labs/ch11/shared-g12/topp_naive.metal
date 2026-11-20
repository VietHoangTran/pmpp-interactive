// Provided for lab 11.6 (do not edit): the Metal counterpart of topp_naive.cu, the benchmark
// baseline. One thread per row walks the row with Fig. 11.1's sequential loop, made
// exclusive (p. 254). Contract as in the lab.
#include <metal_stdlib>
using namespace metal;

kernel void top_p_mask_naive(device const float* probs      [[buffer(0)]],
                             device float*       massBefore [[buffer(1)]],
                             device uchar*       keep       [[buffer(2)]],
                             device int*         kept       [[buffer(3)]],
                             constant uint&      rows       [[buffer(4)]],
                             constant uint&      cols       [[buffer(5)]],
                             constant float&     p          [[buffer(6)]],
                             uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    device const float* in = probs + (ulong)r * cols;
    device float* mass = massBefore + (ulong)r * cols;
    device uchar* k = keep + (ulong)r * cols;
    const bool keepAll = p >= 1.0f;
    float before = 0.0f;
    int count = 0;
    for (uint i = 0; i < cols; ++i) {
        mass[i] = before;
        int keepIt = (i == 0 || keepAll || before < p) ? 1 : 0;
        k[i] = uchar(keepIt);
        count += keepIt;
        before += in[i];
    }
    kept[r] = count;
}
