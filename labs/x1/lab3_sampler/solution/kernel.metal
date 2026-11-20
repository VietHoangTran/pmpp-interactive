// Lab X1.3 reference solution (Metal Shading Language): the kernels no earlier lab has.
// philox_kat, copy_rows, apply_repetition, draw_rows. launch.mm composes them with the earlier
// labs' kernels into encodeSampleRows. The contract and the chain are at the top of kernel.cu.
//
// CUDA                                  Metal
// uint4, make_uint4, uint2              uint4, uint2 (constructors)
// (unsigned long long)a * b             ulong(a) * b   (64-bit integers: Metal 2.2 and later, ch14-facts C3)
// __shfl_down_sync sum                  simd_sum
// warp inclusive scan                   simd_prefix_inclusive_sum
// __ballot_sync, __ffs                  simd_ballot, ctz (after masking to the lanes present)
// expf                                  exp (in the default math mode, ch05-facts C7)
//
// Notes:
//   - MSL has no random number function (notes/today/x1-facts.md C1): Philox is computed here.
//     Random123 itself runs as MSL with mulhi (A4, C4); this file uses the 64-bit product so
//     that it reads like kernel.cu. The known-answer test checks it on this device.
//   - u = (w >> 8) * 2^-24 is exact under any math mode (E2): integer shift, an exact integer to
//     float conversion (round ties to even, C4) and a power-of-two scale.
//   - The weights and sums are float arithmetic under Metal's default math mode, which may also
//     round toward zero (C5): the driver compares tokens only for draws far enough from a
//     boundary (README.md, "Which draws must match").
//   - Thread rank for positions is sg * width + lane: the scan runs in SIMD-group order, and how
//     a threadgroup splits into SIMD-groups is the implementation's (ch04-facts C2).
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256    // threads per threadgroup for draw_rows; launch.mm's kBlock must be equal
#define DRAW_VEC 4        // consecutive weights per thread per chunk

// ---------------------------------------------------------------- 1. Philox4x32-10
// Random123's generator (include/Random123/philox.h; x1-facts A4), written here, not copied;
// Random123's notice is reproduced in kernel.cu.
inline uint4 philox4x32_10(uint4 ctr, uint2 key) {
    for (int round = 0; round < 10; ++round) {
        if (round > 0) {
            key += uint2(0x9E3779B9u, 0xBB67AE85u);
        }
        const ulong p0 = ulong(0xD2511F53u) * ctr.x;
        const ulong p1 = ulong(0xCD9E8D57u) * ctr.z;
        ctr = uint4(uint(p1 >> 32) ^ ctr.y ^ key.x, uint(p1), uint(p0 >> 32) ^ ctr.w ^ key.y, uint(p0));
    }
    return ctr;
}

kernel void philox_kat(device const uint* in    [[buffer(0)]],   // count x (ctr x, y, z, w, key x, y)
                       device uint*       out   [[buffer(1)]],   // count x 4 words
                       constant uint&     count [[buffer(2)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < count) {
        device const uint* a = in + ulong(i) * 6;
        const uint4 w = philox4x32_10(uint4(a[0], a[1], a[2], a[3]), uint2(a[4], a[5]));
        out[ulong(i) * 4 + 0] = w.x;
        out[ulong(i) * 4 + 1] = w.y;
        out[ulong(i) * 4 + 2] = w.z;
        out[ulong(i) * 4 + 3] = w.w;
    }
}

// M5's mapping: counter (step low, step high, stream low, stream high), key (seed low, seed high).
inline uint row_bits(ulong seed, ulong stream, ulong step) {
    return philox4x32_10(uint4(uint(step), uint(step >> 32), uint(stream), uint(stream >> 32)),
                         uint2(uint(seed), uint(seed >> 32))).x;
}

// ---------------------------------------------------------------- 2. u from 32 bits
inline float uniform_from_bits(uint w) {
    return float(w >> 8) * (1.0f / 16777216.0f);
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

// llama.cpp's rule (x1-facts D2): seen tokens, logit <= 0 times the penalty, positive divided.
kernel void apply_repetition(device float*      logits  [[buffer(0)]],
                             device const uint* counts  [[buffer(1)]],
                             constant uint&     n       [[buffer(2)]],
                             constant float&    penalty [[buffer(3)]],
                             uint i [[thread_position_in_grid]]) {
    if (i < n && counts[i] > 0u) {
        const float x = logits[i];
        logits[i] = x <= 0.0f ? x * penalty : x / penalty;
    }
}

// ---------------------------------------------------------------- 4. the draw
#define DRAW_U_FROM_BUFFER 1u   // flags: u comes from u_in (the test entry), not from Philox
#define DRAW_WRITE_U 2u         // flags: write u to u_out

inline simd_vote::vote_t lanes_present(uint width) {
    return width >= 64 ? ~simd_vote::vote_t(0) : (simd_vote::vote_t(1) << width) - 1;
}

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
    threadgroup float sums[BLOCK_SIZE];        // one entry per SIMD-group; BLOCK_SIZE is enough
    threadgroup float chunk_s[2][BLOCK_SIZE];  // SIMD-group totals of a chunk, double-buffered
    threadgroup int hit_s[BLOCK_SIZE];

    const uint me = sg * width + lane;
    const ulong row_start = ulong(row) * stride;
    device const float* l = sorted_logits + row_start;
    const int m = n[row];
    const float l0 = l[0];

    // (a) The total.
    float part = 0.0f;
    for (int i = int(me); i < m; i += int(bdim)) {
        part += exp((l[i] - l0) * inv_t);
    }
    part = simd_sum(part);
    if (lane == 0) {
        sums[sg] = part;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float total = 0.0f;
    for (uint g = 0; g < groups; ++g) {
        total += sums[g];
    }

    // (b) u and the target.
    const float u = (flags & DRAW_U_FROM_BUFFER) != 0u ? u_in[row]
                                                       : uniform_from_bits(row_bits(seeds[row], streams[row], steps[row]));
    const float target = u * total;

    // (c) The chunks; every decision is uniform across the threadgroup.
    int winner = -1;
    float aggregate = 0.0f;
    uint buf = 0;
    for (int base = 0; base < m; base += int(bdim) * DRAW_VEC, buf ^= 1u) {
        const int first = base + int(me) * DRAW_VEC;
        float local[DRAW_VEC];
        bool positive[DRAW_VEC];
        float run = 0.0f;
        for (int j = 0; j < DRAW_VEC; ++j) {
            const float w = first + j < m ? exp((l[first + j] - l0) * inv_t) : 0.0f;
            positive[j] = w > 0.0f;
            run += w;
            local[j] = run;
        }
        const float incl = simd_prefix_inclusive_sum(run);
        if (lane == width - 1) {
            chunk_s[buf][sg] = incl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float carry = 0.0f, chunk = 0.0f;
        for (uint g = 0; g < groups; ++g) {
            const float s = chunk_s[buf][g];
            carry += g < sg ? s : 0.0f;
            chunk += s;
        }
        if (aggregate + chunk > target) {
            const float before = aggregate + carry + (incl - run);
            int hit = -1;
            for (int j = DRAW_VEC - 1; j >= 0; --j) {
                if (first + j < m && positive[j] && before + local[j] > target) {
                    hit = j;
                }
            }
            const simd_vote::vote_t vote = simd_vote::vote_t(simd_ballot(hit >= 0)) & lanes_present(width);
            if (vote == 0) {
                if (lane == 0) {
                    hit_s[sg] = -1;
                }
            } else if (lane == uint(ctz(vote))) {
                hit_s[sg] = first + hit;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (uint g = 0; g < groups && winner < 0; ++g) {
                winner = hit_s[g];
            }
            if (winner >= 0) {
                break;
            }
        }
        aggregate += chunk;
    }

    // (d) The fallback: the last positive weight.
    if (me == 0) {
        if (winner < 0) {
            winner = 0;
            for (int i = m - 1; i > 0; --i) {
                if (exp((l[i] - l0) * inv_t) > 0.0f) {
                    winner = i;
                    break;
                }
            }
        }
        tokens[row] = cand_idx[row_start + ulong(sorted_pos[row_start + ulong(winner)])];
        if ((flags & DRAW_WRITE_U) != 0u) {
            u_out[row] = u;
        }
    }
}
