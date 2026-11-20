// PROVIDED, do not edit. A copy of labs/x1/lab1_topk/solution/kernel.metal (lab X1.1: top-k by radix select),
// used by lab X1.3. Unchanged; app/src/chapters/x1/copies-g11.test.ts checks that it stays so.
// Lab X1.1 reference solution (Metal Shading Language). The same two kernels as kernel.cu:
// topk_threshold and topk_gather, one threadgroup of BLOCK_SIZE threads per row; launch.mm
// encodes both. The contract is the one at the top of kernel.cu.
//
// CUDA                                  Metal
// __shared__ unsigned int a[N]          threadgroup uint a[N]   (declared in the kernel)
// cuda::atomic_ref on shared memory     threadgroup atomic_uint a[N]; atomic_fetch_add_explicit
// __syncthreads()                       threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize, laneIdx(), warpIdx()        [[threads_per_simdgroup]], [[thread_index_in_simdgroup]],
//                                       [[simdgroup_index_in_threadgroup]]
// warp scan with __shfl_up_sync         simd_prefix_inclusive_sum
// __shfl_sync(m, v, w - 1)              simd_shuffle(v, w - 1)
// __ballot_sync, __popc, __ffs          simd_ballot, popcount, ctz
// (const unsigned int*)logits           device const uint* logits: the buffer holds float32
//
// Notes:
//   - The logits are bound as uint. Both of Metal's candidate default math modes allow "no
//     signed zeros" (notes/today/ch14-facts.md C3), so the -0.0 step is an integer test.
//   - MSL has no atomic view of a plain object: the histogram is atomic_uint, read with
//     atomic_load_explicit (ch09-facts C1).
//   - simd_ballot's vote is 64 bits whatever the SIMD-group width, and its bits above the width
//     are undefined (ch12-facts C2). The rank popcount(vote & lanes below) never reads them; a
//     SIMD-group's count comes from simd_sum; the first set lane is taken after masking the
//     vote to the lanes that exist.
//   - How a threadgroup splits into SIMD-groups is the implementation's (ch04-facts C2). Ranks
//     run in (SIMD-group, lane) order, so a thread's column comes from me = sg * width + lane,
//     not from thread_position_in_threadgroup, as in lab 14.3.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256                // threads per threadgroup; launch.mm's kBlock must be equal
#define DIGIT_BITS 8
#define BINS (1u << DIGIT_BITS)
#define SCRATCH_PER_ROW 4             // K*, ties still needed, two unused

// Lab 14.3's key, ascending (kernel.cu says more).
inline uint ordered_key(uint u) {
    if (u == 0x80000000u) {
        u = 0u;
    }
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

// The lanes that exist in a SIMD-group of `width` lanes, as bits of a 64-bit vote.
inline simd_vote::vote_t lanes_present(uint width) {
    return width >= 64 ? ~simd_vote::vote_t(0) : (simd_vote::vote_t(1) << width) - 1;
}

// ---------------------------------------------------------------- 1. the threshold
kernel void topk_threshold(device const uint* logits  [[buffer(0)]],   // the float32 bits
                           device uint*       scratch [[buffer(1)]],
                           constant uint&     cols    [[buffer(2)]],
                           constant uint&     k       [[buffer(3)]],
                           uint row   [[threadgroup_position_in_grid]],
                           uint bdim  [[threads_per_threadgroup]],
                           uint width [[threads_per_simdgroup]],
                           uint lane  [[thread_index_in_simdgroup]],
                           uint sg    [[simdgroup_index_in_threadgroup]]) {
    threadgroup atomic_uint hist[BINS];
    threadgroup uint pick[2];            // the crossing bin, and the keys in the bins above it

    const uint me = sg * width + lane;   // this thread's rank in SIMD-group order
    const uint n = cols;
    device const uint* in = logits + ulong(row) * n;

    uint prefix = 0, mask = 0;
    uint need = min(k, n);

    for (int shift = 32 - DIGIT_BITS; shift >= 0; shift -= DIGIT_BITS) {
        for (uint b = me; b < BINS; b += bdim) {
            atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = me; c < n; c += bdim) {
            const uint key = ordered_key(in[c]);
            if ((key & mask) == prefix) {
                atomic_fetch_add_explicit(&hist[(key >> shift) & (BINS - 1u)], 1u, memory_order_relaxed);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // The bin walk, from the top, in SIMD-group 0.
        if (sg == 0) {
            uint above = 0;
            for (uint base = 0; base < BINS; base += width) {
                const uint b = BINS - 1u - (base + lane);
                const uint count = atomic_load_explicit(&hist[b], memory_order_relaxed);
                const uint incl = simd_prefix_inclusive_sum(count);
                const simd_vote::vote_t reached =
                    simd_vote::vote_t(simd_ballot(above + incl >= need)) & lanes_present(width);
                if (reached != 0) {
                    if (lane == uint(ctz(reached))) {
                        pick[0] = b;
                        pick[1] = above + incl - count;
                    }
                    break;
                }
                above += simd_shuffle(incl, ushort(width - 1));
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const uint bin = pick[0];
        const uint in_bin = atomic_load_explicit(&hist[bin], memory_order_relaxed);
        need -= pick[1];
        prefix |= bin << shift;
        mask |= (BINS - 1u) << shift;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (need == in_bin) {
            break;
        }
    }

    if (me == 0) {
        device uint* out = scratch + ulong(row) * SCRATCH_PER_ROW;
        out[0] = prefix;   // K*
        out[1] = need;     // ties at K* still needed
        out[2] = 0u;
        out[3] = 0u;
    }
}

// ---------------------------------------------------------------- 2. the gather
kernel void topk_gather(device const uint* logits  [[buffer(0)]],   // the float32 bits
                        device const uint* scratch [[buffer(1)]],
                        device int*        out_idx [[buffer(2)]],
                        device uint*       out_val [[buffer(3)]],   // outVal, as bits
                        constant uint&     cols    [[buffer(4)]],
                        constant uint&     k       [[buffer(5)]],
                        uint row    [[threadgroup_position_in_grid]],
                        uint bdim   [[threads_per_threadgroup]],
                        uint width  [[threads_per_simdgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint sg     [[simdgroup_index_in_threadgroup]],
                        uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup uint group_ties[BLOCK_SIZE];    // one entry per SIMD-group; BLOCK_SIZE is enough
    threadgroup uint group_keeps[BLOCK_SIZE];

    const uint me = sg * width + lane;
    const simd_vote::vote_t lanes_below = (simd_vote::vote_t(1) << lane) - 1;
    const uint n = cols;
    device const uint* in = logits + ulong(row) * n;
    device int* idx_row = out_idx + ulong(row) * k;
    device uint* val_row = out_val + ulong(row) * k;
    device const uint* thr = scratch + ulong(row) * SCRATCH_PER_ROW;
    const uint k_star = thr[0], need = thr[1];
    const uint total = min(k, n);

    uint ties_before = 0, kept_before = 0;
    for (uint base = 0; base < n && kept_before < total; base += bdim) {
        const uint c = base + me;
        uint u = 0;
        bool above = false, tie = false;
        if (c < n) {
            u = in[c];
            const uint key = ordered_key(u);
            above = key > k_star;
            tie = key == k_star;
        }

        const simd_vote::vote_t tie_vote = simd_vote::vote_t(simd_ballot(tie));
        const uint group_tie_count = simd_sum(tie ? 1u : 0u);
        if (lane == 0) {
            group_ties[sg] = group_tie_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint tie_offset = 0, round_ties = 0;
        for (uint g = 0; g < groups; ++g) {
            const uint t = group_ties[g];
            tie_offset += g < sg ? t : 0u;
            round_ties += t;
        }
        const bool keep = above || (tie && ties_before + tie_offset + uint(popcount(tie_vote & lanes_below)) < need);

        const simd_vote::vote_t keep_vote = simd_vote::vote_t(simd_ballot(keep));
        const uint group_keep_count = simd_sum(keep ? 1u : 0u);
        if (lane == 0) {
            group_keeps[sg] = group_keep_count;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint keep_offset = 0, round_kept = 0;
        for (uint g = 0; g < groups; ++g) {
            const uint t = group_keeps[g];
            keep_offset += g < sg ? t : 0u;
            round_kept += t;
        }
        if (keep) {
            const uint slot = kept_before + keep_offset + uint(popcount(keep_vote & lanes_below));
            idx_row[slot] = int(c);
            val_row[slot] = u;
        }
        ties_before += round_ties;
        kept_before += round_kept;
    }
}
