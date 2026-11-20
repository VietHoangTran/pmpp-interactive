// Lab X2.2 reference solution (Metal Shading Language). The same Q4_0 matrix-vector product
// as kernel.cu, on llama.cpp's block exactly as llama.cpp declares it on Metal (ggml-common.h
// with GGML_COMMON_DECL_METAL, where ggml_half is half; notes/today/x2-facts.md C5, D1).
//   Y[b][j] = sum over k of X[b][k] * w[j][k],  w = (code - 8) * d,  1 <= batch <= MAX_BATCH.
//
// CUDA                               Metal
// warp                               SIMD-group
// warpSize                           [[threads_per_simdgroup]]
// warpSum (shuffles and a mask)      simd_sum: the sum, given to every lane, no mask
// uint16_t d + halfBitsToFloat       half d, converted with float(...)
// two 16-bit loads of qs             four uchar loads of qs
//
// half has size 2 and alignment 2, so the block is 18 bytes with alignment 2 and every other
// block's qs starts at 2 mod 4 (x2-facts C5): no uint or uchar4 loads of qs. llama.cpp reads
// qs as 16-bit words through a pointer cast (C5); the MSL specification describes no pointer
// casts (infra-facts C7), so this kernel reads bytes, which needs none. X is read as float4,
// Metal's documented vector type (infra-facts C7): each lane's 4 low-nibble activations start
// at a multiple of 4 floats. simd_sum needs an Apple7 GPU or later (infra-facts C2).
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8
#define QK4_0 32             // weights per Q4_0 block (llama.cpp's name)
#define LANES_PER_BLOCK 4    // lanes sharing one block: 4 bytes of qs each

struct block_q4_0 {
    half d;                  // the scale
    uchar qs[QK4_0 / 2];     // byte j: code j (low nibble), code j + 16 (high nibble)
};

kernel void gemv_q4_0(device const float4*     X     [[buffer(0)]],   // [batch x in/4]
                      device const block_q4_0* W     [[buffer(1)]],   // [out x in/32]
                      device float*            Y     [[buffer(2)]],   // [batch x out]
                      constant uint&           batch [[buffer(3)]],
                      constant uint&           in    [[buffer(4)]],
                      constant uint&           out   [[buffer(5)]],
                      uint tg     [[threadgroup_position_in_grid]],
                      uint width  [[threads_per_simdgroup]],
                      uint lane   [[thread_index_in_simdgroup]],
                      uint sg     [[simdgroup_index_in_threadgroup]],
                      uint groups [[simdgroups_per_threadgroup]]) {
    const uint j = tg * groups + sg;    // this SIMD-group's output
    if (j >= out) {
        return;                         // the whole SIMD-group leaves together
    }
    const uint nb = in / QK4_0;
    device const block_q4_0* row = W + j * nb;
    const uint s = lane % LANES_PER_BLOCK;          // which 4 bytes of qs this lane reads
    const uint blocksPerStep = width / LANES_PER_BLOCK;
    const uint xRow = in / 4;                       // float4 elements per row of X

    float sum[MAX_BATCH];
    for (uint b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    for (uint i = lane / LANES_PER_BLOCK; i < nb; i += blocksPerStep) {
        device const block_q4_0& blk = row[i];
        const float d = float(blk.d);
        const uchar b0 = blk.qs[4 * s], b1 = blk.qs[4 * s + 1], b2 = blk.qs[4 * s + 2], b3 = blk.qs[4 * s + 3];
        const float4 lo = float4(b0 & 0x0F, b1 & 0x0F, b2 & 0x0F, b3 & 0x0F);   // weights 4s .. 4s + 3
        const float4 hi = float4(b0 >> 4, b1 >> 4, b2 >> 4, b3 >> 4);           // weights 16 + 4s ..
        for (uint b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                const float4 xl = X[b * xRow + i * (QK4_0 / 4) + s];       // floats 4s .. 4s + 3 of the block
                const float4 xh = X[b * xRow + i * (QK4_0 / 4) + 4 + s];   // floats 16 + 4s ..
                const float qx = dot(lo, xl) + dot(hi, xh);
                const float sx = dot(xl + xh, float4(1.0f));
                sum[b] += d * (qx - 8.0f * sx);     // the -8 once per lane's share of the block
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
