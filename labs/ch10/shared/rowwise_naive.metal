// Provided for lab 10.3 (do not edit): the benchmark baseline, one thread per row.
// The Metal counterpart of rowwise_naive.cu; see the comment there.
#include <metal_stdlib>
using namespace metal;

kernel void softmax_rows_naive(device const float* x [[buffer(0)]],
                               device float*       y [[buffer(1)]],
                               constant uint&   rows [[buffer(2)]],
                               constant uint&   cols [[buffer(3)]],
                               constant float& scale [[buffer(4)]],
                               uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    device const float* in = x + ulong(r) * cols;
    device float* out = y + ulong(r) * cols;
    float m = -FLT_MAX;
    for (uint c = 0; c < cols; ++c) {
        float v = scale * in[c];
        m = v > m ? v : m;
    }
    float s = 0.0f;
    for (uint c = 0; c < cols; ++c) {
        s += exp(scale * in[c] - m);
    }
    float inv = 1.0f / s;
    for (uint c = 0; c < cols; ++c) {
        out[c] = exp(scale * in[c] - m) * inv;
    }
}

kernel void argmax_rows_naive(device const float* x [[buffer(0)]],
                              device int*       idx [[buffer(1)]],
                              constant uint&   rows [[buffer(2)]],
                              constant uint&   cols [[buffer(3)]],
                              uint r [[thread_position_in_grid]]) {
    if (r >= rows) return;
    device const float* in = x + ulong(r) * cols;
    float best = in[0];
    int at = 0;
    for (uint c = 1; c < cols; ++c) {
        if (in[c] > best) {   // strictly greater: the first maximum stays
            best = in[c];
            at = int(c);
        }
    }
    idx[r] = at;
}
