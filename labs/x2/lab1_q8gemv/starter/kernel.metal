// Lab X2.1 starter (Metal Shading Language). The same Q8_0 matrix-vector product as
// kernel.cu:
//   Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k],   1 <= batch <= MAX_BATCH,
// codes Wq [out x in] as char4 (4 codes per element), scales Wd [out x in/32] as half.
// Not in the book; the format is llama.cpp's Q8_0 with codes and scales in two arrays
// (notes/today/x2-facts.md D1, D5).
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
// char4 has size 4 and alignment 4 in MSL (infra-facts C9) and half is a built-in type, so
// Metal needs neither of kernel.cu's bit helpers. X is declared float4, Metal's documented
// way to read four floats at once (infra-facts C7). simd_sum needs an Apple7 GPU or later
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
    // TODO 4: the same kernel as gemvQ8_0Kernel in kernel.cu, one SIMD-group per output.
    //   a. This SIMD-group's output j comes from tg, groups and sg. Past the end: return.
    //   b. Row j's codes and scales: where do they start, in char4 and in half elements?
    //   c. The lanes stride by `width` along row j's char4 elements. Which block, and so
    //      which scale, does element w belong to? Element w of row b of X (as float4) holds
    //      the four activations that meet it.
    //   d. Accumulate in float. dot(float4, float4) returns float; what would dot on half4
    //      return?
    //   e. simd_sum each sum (every lane must reach it); lane 0 writes Y[b * out + j].
}
