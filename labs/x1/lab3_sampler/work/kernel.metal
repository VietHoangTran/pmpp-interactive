// Lab X1.3: the Metal kernels no earlier lab has: philox_kat, copy_rows (provided),
// apply_repetition, draw_rows. launch.mm composes them with the earlier labs' kernels into
// encodeSampleRows. README.md has the contract and the chain; kernel.cu's TODOs say what each
// piece computes.
//
// CUDA                                  Metal
// uint4, make_uint4, uint2              uint4, uint2 (constructors)
// (unsigned long long)a * b             ulong(a) * b   (64-bit integers: Metal 2.2 and later, ch14-facts C3)
// __shfl_down_sync sum                  simd_sum
// warp inclusive scan                   simd_prefix_inclusive_sum
// __ballot_sync, __ffs                  simd_ballot, ctz (mask the vote to the lanes present first)
// expf                                  exp
//
// Things Metal makes you decide (README.md, "Portability"):
//   - MSL has no random number function (notes/today/x1-facts.md C1): you compute Philox here.
//     mulhi gives the high half of a 32 x 32 product too (C4).
//   - simd_ballot's vote is 64 bits whatever the SIMD-group width, and its bits above the width
//     are undefined (ch12-facts C2).
//   - The scan runs in (SIMD-group, lane) order: give each thread its positions from
//     sg * width + lane (ch04-facts C2).
//   - Float sums here are under Metal's default math mode, which may also round toward zero
//     (x1-facts C5). The driver compares tokens only where that cannot matter.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256    // threads per threadgroup for draw_rows; launch.mm's kBlock must be equal
#define DRAW_VEC 4

// ---------------------------------------------------------------- 1. Philox4x32-10
inline uint4 philox4x32_10(uint4 ctr, uint2 key) {
    // TODO: kernel.cu's TODO 1.
    return ctr;
}

kernel void philox_kat(device const uint* in    [[buffer(0)]],   // count x (ctr x, y, z, w, key x, y)
                       device uint*       out   [[buffer(1)]],   // count x 4 words
                       constant uint&     count [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    // TODO: kernel.cu's TODO 2.
}

// ---------------------------------------------------------------- 2. u from 32 bits
inline float uniform_from_bits(uint w) {
    // TODO: kernel.cu's TODO 3 (and the mapping from seed, stream and step).
    return 0.0f;
}

// ---------------------------------------------------------------- 3. copy and repetition
// Provided: the workspace copy.
kernel void copy_rows(device const float* in  [[buffer(0)]],
                      device float*       out [[buffer(1)]],
                      constant uint&      n   [[buffer(2)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) {
        out[i] = in[i];
    }
}

kernel void apply_repetition(device float*      logits  [[buffer(0)]],
                             device const uint* counts  [[buffer(1)]],
                             constant uint&     n       [[buffer(2)]],
                             constant float&    penalty [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    // TODO: kernel.cu's TODO 4.
}

// ---------------------------------------------------------------- 4. the draw
#define DRAW_U_FROM_BUFFER 1u   // flags: u comes from u_in (the test entry), not from Philox
#define DRAW_WRITE_U 2u         // flags: write u to u_out

kernel void draw_rows(device const float* sorted_logits [[buffer(0)]],
                      device const int*   sorted_pos    [[buffer(1)]],
                      device const int*   cand_idx      [[buffer(2)]],
                      device const int*   n             [[buffer(3)]],
                      device const ulong* seeds         [[buffer(4)]],
                      device const ulong* streams       [[buffer(5)]],
                      device const ulong* steps         [[buffer(6)]],
                      device const float* u_in          [[buffer(7)]],
                      device int*         tokens        [[buffer(8)]],
                      device float*       u_out         [[buffer(9)]],
                      constant uint&      stride        [[buffer(10)]],
                      constant float&     inv_t         [[buffer(11)]],
                      constant uint&      flags         [[buffer(12)]],
                      uint row    [[threadgroup_position_in_grid]],
                      uint bdim   [[threads_per_threadgroup]],
                      uint width  [[threads_per_simdgroup]],
                      uint lane   [[thread_index_in_simdgroup]],
                      uint sg     [[simdgroup_index_in_threadgroup]],
                      uint groups [[simdgroups_per_threadgroup]]) {
    // TODO: kernel.cu's TODOs 5 to 8. Per-SIMD-group values can live in threadgroup arrays of
    //       BLOCK_SIZE entries, enough for any width.
}
