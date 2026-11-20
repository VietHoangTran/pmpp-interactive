// Lab X2.2 starter (Metal Shading Language). The same Q4_0 matrix-vector product as
// kernel.cu, on llama.cpp's block exactly as llama.cpp declares it on Metal (ggml-common.h
// with GGML_COMMON_DECL_METAL, where ggml_half is half; notes/today/x2-facts.md C5, D1).
//   Y[b][j] = sum over k of X[b][k] * w[j][k],  w = (code - 8) * d,  1 <= batch <= MAX_BATCH.
//
// CUDA                               Metal
// warp                               SIMD-group
// warpSize                           [[threads_per_simdgroup]]
// warpSum (shuffles and a mask)      simd_sum: the sum, given to every lane, no mask
// uint16_t d + halfBitsToFloat       half d, converted with float(...)
//
// half has size 2 and alignment 2, so this struct is 18 bytes with alignment 2 (x2-facts C5),
// the same trap as in kernel.cu. X is declared float4, Metal's documented way to read four
// floats at once (infra-facts C7). simd_sum needs an Apple7 GPU or later (infra-facts C2).
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8
#define QK4_0 32             // weights per Q4_0 block (llama.cpp's name)

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
    // TODO 3: the same kernel as gemvQ4_0Kernel in kernel.cu, one SIMD-group per output.
    //   a. This SIMD-group's output j comes from tg, groups and sg. Past the end: return.
    //   b. The lanes share row j's blocks the way you chose in kernel.cu, with `width` for
    //      the SIMD-group width.
    //   c. Read qs as uchar: no uint or uchar4 loads from an 18-byte block. Which float4
    //      elements of X row b do the low and the high nibbles meet?
    //   d. Accumulate in float; dot(float4, float4) returns float.
    //   e. simd_sum each sum (every lane must reach it); lane 0 writes Y[b * out + j].
}
