// Provided for lab 12.4 (do not edit): the benchmark baseline, one thread per row.
// The Metal counterpart of rowwise_serial.cu; see the comment there.
#include <metal_stdlib>
using namespace metal;

kernel void compact_serial(device const float* logits    [[buffer(0)]],
                           device const float* threshold [[buffer(1)]],
                           device int*         outIdx    [[buffer(2)]],
                           device float*       outVal    [[buffer(3)]],
                           device uint*        counts    [[buffer(4)]],
                           constant uint&      rows      [[buffer(5)]],
                           constant uint&      cols      [[buffer(6)]],
                           uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    const ulong rowStart = ulong(r) * cols;
    device const float* in = logits + rowStart;
    const float thr = threshold[r];
    uint n = 0;
    for (uint c = 0; c < cols; ++c) {
        const float v = in[c];
        if (v >= thr) {   // llama.cpp: if (logit >= min_logit) push_back
            outIdx[rowStart + n] = int(c);
            outVal[rowStart + n] = v;
            ++n;
        }
    }
    counts[r] = n;
}
