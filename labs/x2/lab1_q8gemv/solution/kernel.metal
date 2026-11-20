// Lab X2.1 reference solution (Metal Shading Language). The same Q8_0 matrix-vector product
// as kernel.cu:
//   Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k],   1 <= batch <= MAX_BATCH,
// codes Wq [out x in] as char4 (4 codes per element), scales Wd [out x in/32] as half.
// Not in the book; the format is llama.cpp's Q8_0 with codes and scales in two arrays
// (notes/today/x2-facts.md D1, D5; kernel.cu says what is whose).
//
// CUDA                               Metal
// warp                               SIMD-group
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// warpSum (shuffles and a mask)      simd_sum: the sum, given to every lane, no mask
// uint32_t word + byte extraction    char4, converted with float4(...)
// uint16_t bits + halfBitsToFloat    half, converted with float(...)
//
// char4 has size 4 and alignment 4 in MSL (infra-facts C9; x2-facts table "Where the backends
// really differ"), and half is a built-in type (C5), so Metal needs neither of kernel.cu's bit
// helpers. X is read as float4, Metal's documented vector type, rather than through a cast
// (infra-facts C7). dot(float4, float4) returns float; dot on half4 would return half (x2-facts
// C1), the narrow accumulator appendix A warns about. simd_sum needs an Apple7 GPU or later
// (infra-facts C2).
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8
#define QK8_0 32            // weights per Q8_0 block (llama.cpp's name)
#define CODES_PER_WORD 4    // codes in one char4

kernel void gemv_q8_0(device const float4* X     [[buffer(0)]],   // [batch x in/4]
                      device const char4*  Wq    [[buffer(1)]],   // [out x in/4]
                      device const half*   Wd    [[buffer(2)]],   // [out x in/32]
                      device float*        Y     [[buffer(3)]],   // [batch x out]
                      constant uint&       batch [[buffer(4)]],
                      constant uint&       in    [[buffer(5)]],
                      constant uint&       out   [[buffer(6)]],
                      uint tg     [[threadgroup_position_in_grid]],
                      uint width  [[threads_per_simdgroup]],
                      uint lane   [[thread_index_in_simdgroup]],
                      uint sg     [[simdgroup_index_in_threadgroup]],
                      uint groups [[simdgroups_per_threadgroup]]) {
    const uint j = tg * groups + sg;    // this SIMD-group's output
    if (j >= out) {
        return;                         // the whole SIMD-group leaves together
    }
    const uint words = in / CODES_PER_WORD;
    device const char4* q = Wq + j * words;          // row j's codes
    device const half* d = Wd + j * (in / QK8_0);    // row j's scales

    float sum[MAX_BATCH];
    for (uint b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    // Element w of row j holds codes 4w .. 4w + 3, all in block w / 8.
    for (uint w = lane; w < words; w += width) {
        const float4 qv = float4(q[w]);
        const float scale = float(d[w / (QK8_0 / CODES_PER_WORD)]);
        for (uint b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                sum[b] += scale * dot(X[b * words + w], qv);   // one scale for the four products
            }
        }
    }
    for (uint b = 0; b < MAX_BATCH; ++b) {
        if (b < batch) {                // the same for every lane, so every lane reaches simd_sum
            const float total = simd_sum(sum[b]);
            if (lane == 0) {
                Y[b * out + j] = total; // nobody else writes this element
            }
        }
    }
}
