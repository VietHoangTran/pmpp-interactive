// Labs X2.3 and X2.4, provided (do not edit). A STAND-IN for lab X2.2's Metal kernel, used
// only when lab X2.2's solution is absent: the same fused float Q4_0 GEMV as
// gemv_q4_0_g12.cu, one SIMD-group per output, simd_sum for the reduction.
//   Y[batch x out] = X[batch x in] * W^T, W as (out, in) Q4_0 blocks, batch <= 8, in % 32 == 0.
// Not in the book, and ours. The block is llama.cpp's block_q4_0 (x2-facts D1): a half
// scale and 16 bytes, 2-byte aligned, so the codes are read as 16-bit words (C5).
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8

struct block_q4_0 {
    ushort d;         // FP16 bits of the scale
    uchar qs[16];     // byte j: weight j (low nibble), weight j + 16 (high nibble)
};

kernel void gemv_q4_0(device const float*      X     [[buffer(0)]],
                      device const block_q4_0* W     [[buffer(1)]],
                      device float*            Y     [[buffer(2)]],
                      constant uint&           batch [[buffer(3)]],
                      constant uint&           in    [[buffer(4)]],
                      constant uint&           out   [[buffer(5)]],
                      uint tg     [[threadgroup_position_in_grid]],
                      uint width  [[threads_per_simdgroup]],
                      uint lane   [[thread_index_in_simdgroup]],
                      uint sg     [[simdgroup_index_in_threadgroup]],
                      uint groups [[simdgroups_per_threadgroup]]) {
    const uint j = tg * groups + sg;
    if (j >= out) {
        return;
    }
    const uint nb = in / 32;
    device const block_q4_0* row = W + j * nb;
    float sum[MAX_BATCH];
    for (uint b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    for (uint kb = lane; kb < nb; kb += width) {
        const float d = float(as_type<half>(row[kb].d));
        device const ushort* q16 = (device const ushort*)row[kb].qs;
        float acc[MAX_BATCH], xs[MAX_BATCH];
        for (uint b = 0; b < MAX_BATCH; ++b) {
            acc[b] = 0.0f;
            xs[b] = 0.0f;
        }
        for (uint i = 0; i < 8; ++i) {
            const uint u = q16[i];
            const float c0 = float(u & 0xFu), c1 = float((u >> 8) & 0xFu);
            const float c2 = float((u >> 4) & 0xFu), c3 = float((u >> 12) & 0xFu);
            for (uint b = 0; b < MAX_BATCH; ++b) {
                if (b < batch) {
                    device const float* x = X + b * in + kb * 32;
                    acc[b] += c0 * x[2 * i] + c1 * x[2 * i + 1] + c2 * x[2 * i + 16] + c3 * x[2 * i + 17];
                    xs[b] += x[2 * i] + x[2 * i + 1] + x[2 * i + 16] + x[2 * i + 17];
                }
            }
        }
        for (uint b = 0; b < MAX_BATCH; ++b) {
            sum[b] += d * (acc[b] - 8.0f * xs[b]);
        }
    }
    for (uint b = 0; b < MAX_BATCH; ++b) {
        const float t = (b < batch) ? simd_sum(sum[b]) : 0.0f;
        if (lane == 0 && b < batch) {
            Y[b * out + j] = t;
        }
    }
}
