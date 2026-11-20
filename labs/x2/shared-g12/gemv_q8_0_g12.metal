// Lab X2.4, provided (do not edit). A STAND-IN for lab X2.1's Metal kernel, used only when
// lab X2.1's solution is absent: the fused float Q8_0 GEMV of gemv_q8_0_g12.cu, one
// SIMD-group per output, simd_sum for the reduction.
//   Y[batch x out] = X[batch x in] * W^T; Wq: out * in int8 codes; Wd: out * in / 32 FP16
//   scales. batch <= 8, in % 32 == 0. Not in the book, and ours.
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8

kernel void gemv_q8_0(device const float*  X     [[buffer(0)]],
                      device const char4*  Wq    [[buffer(1)]],   // the codes, four at a time
                      device const ushort* Wd    [[buffer(2)]],   // FP16 bits of the scales
                      device float*        Y     [[buffer(3)]],
                      constant uint&       batch [[buffer(4)]],
                      constant uint&       in    [[buffer(5)]],
                      constant uint&       out   [[buffer(6)]],
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
    float sum[MAX_BATCH];
    for (uint b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    for (uint kb = lane; kb < nb; kb += width) {
        const float d = float(as_type<half>(Wd[j * nb + kb]));
        device const char4* q = Wq + (j * in + kb * 32) / 4;    // in % 32 == 0: a whole char4 boundary
        float acc[MAX_BATCH];
        for (uint b = 0; b < MAX_BATCH; ++b) {
            acc[b] = 0.0f;
        }
        for (uint i = 0; i < 8; ++i) {
            const float4 c = float4(q[i]);
            for (uint b = 0; b < MAX_BATCH; ++b) {
                if (b < batch) {
                    device const float* x = X + b * in + kb * 32 + 4 * i;
                    acc[b] += c.x * x[0] + c.y * x[1] + c.z * x[2] + c.w * x[3];
                }
            }
        }
        for (uint b = 0; b < MAX_BATCH; ++b) {
            sum[b] += d * acc[b];
        }
    }
    for (uint b = 0; b < MAX_BATCH; ++b) {
        const float t = (b < batch) ? simd_sum(sum[b]) : 0.0f;
        if (lane == 0 && b < batch) {
            Y[b * out + j] = t;
        }
    }
}
