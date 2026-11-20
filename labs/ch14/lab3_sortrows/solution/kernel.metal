// Lab 14.3 reference solution (Metal Shading Language). The same three kernels as kernel.cu:
// desc_keys, sort_rows_pass, gather_rows. One threadgroup of BLOCK_SIZE threads per row;
// launch.mm encodes the whole sort. The contract is the one at the top of kernel.cu.
//
// CUDA                                 Metal
// __shared__ unsigned int a[N]         threadgroup uint a[N]   (declared in the kernel)
// cuda::atomic_ref<...> on shared      threadgroup atomic_uint a[N]; atomic_fetch_add_explicit
// __syncthreads()                      threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                             [[threads_per_simdgroup]]
// warp scan with __shfl_up_sync        simd_prefix_inclusive_sum(v)
// (const unsigned int*)probs           device const uint* probs: the host buffer holds float32
//
// Notes:
//   - probs and sortedProbs are bound as uint: the kernels copy and compare bits and never
//     perform a float operation. Both of Metal's candidate default math modes (fast and
//     relaxed) allow "no signed zeros", and fast math leaves NaN and INF undefined
//     (notes/today/ch14-facts.md C3), so a float-side test for -0.0 or NaN could be rewritten.
//     On uint nothing can be.
//   - MSL has no atomic view of a plain object; the histogram is declared atomic_uint and
//     read with atomic_load_explicit (notes/today/ch09-facts.md C1).
//   - How a threadgroup is split into SIMD-groups is the implementation's (ch04-facts C2). The
//     scan runs in SIMD-group order, so sort_rows_pass ranks its threads by
//     (SIMD-group, lane) and uses that rank, not thread_position_in_threadgroup, for positions.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256                          // threads per threadgroup; one per row
#define RADIX_BITS 8                            // r; launch.mm's kRadixBits must be equal
#define RADIX (1u << RADIX_BITS)
#define KEYS_PER_THREAD 8                       // c
#define TILE (BLOCK_SIZE * KEYS_PER_THREAD)

static_assert(32 % RADIX_BITS == 0, "the passes must cover the 32 key bits exactly");
static_assert(RADIX <= BLOCK_SIZE, "the bucket scan gives each thread at most one bucket");

// ---------------------------------------------------------------- provided: block scan
// Chapter 11's scan-scan-add on unsigned counts, with Metal's SIMD-group prefix sum in place
// of the shuffle loop (infra-facts C2). Returns the sum of v over the threads ranked before
// this one, where the rank is sg * width + lane, and the threadgroup's total in `total`.
// Every thread must call it. group_sums has one entry per SIMD-group (launch.mm sizes it,
// and checks that there are no more SIMD-groups than lanes). Two calls in a row need a
// barrier between them: the second writes group_sums while the first may still read it.
// One difference from Fig. 11.9 (ours): every SIMD-group scans the group totals itself, so
// the second barrier goes. On this lab's own measurements that was the fastest of the
// variants we tried; the scan runs once per key bit for every tile.
inline uint block_exclusive_scan(uint v, threadgroup uint* group_sums, uint lane, uint sg,
                                 uint width, uint groups, thread uint& total) {
    const uint incl = simd_prefix_inclusive_sum(v);
    if (lane == width - 1) {
        group_sums[sg] = incl;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint s = lane < groups ? group_sums[lane] : 0u;
    const uint excl = simd_prefix_exclusive_sum(s);
    total = simd_sum(s);
    return incl - v + simd_shuffle(excl, ushort(sg));
}

// ---------------------------------------------------------------- 1. keys
kernel void desc_keys(device const uint* probs [[buffer(0)]],   // the float32 bits
                      device uint*       keys  [[buffer(1)]],
                      device uint*       idx   [[buffer(2)]],
                      constant uint&     cols  [[buffer(3)]],
                      uint t    [[thread_position_in_threadgroup]],
                      uint row  [[threadgroup_position_in_grid]],
                      uint bdim [[threads_per_threadgroup]]) {
    const ulong base = ulong(row) * cols;
    for (uint c = t; c < cols; c += bdim) {
        uint u = probs[base + c];
        if (u == 0x80000000u) {
            u = 0u;                                  // -0.0 becomes +0.0
        }
        const uint key = (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
        keys[base + c] = ~key;                       // descending
        idx[base + c] = c;
    }
}

// ---------------------------------------------------------------- 2. one digit
inline uint digit_of(uint key, uint shift) {
    return (key >> shift) & (RADIX - 1u);
}

kernel void sort_rows_pass(device const uint* keys_in  [[buffer(0)]],
                           device const uint* idx_in   [[buffer(1)]],
                           device uint*       keys_out [[buffer(2)]],
                           device uint*       idx_out  [[buffer(3)]],
                           constant uint&     cols     [[buffer(4)]],
                           constant uint&     shift    [[buffer(5)]],
                           threadgroup uint*  group_sums [[threadgroup(0)]],
                           uint row    [[threadgroup_position_in_grid]],
                           uint bdim   [[threads_per_threadgroup]],
                           uint width  [[threads_per_simdgroup]],
                           uint lane   [[thread_index_in_simdgroup]],
                           uint sg     [[simdgroup_index_in_threadgroup]],
                           uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup atomic_uint hist[RADIX];
    threadgroup uint offset[RADIX];        // each bucket's next free slot in the output row
    threadgroup uint tile_start[RADIX];    // where each bucket starts in the sorted tile
    threadgroup uint keys_s[TILE];
    threadgroup uint idx_s[TILE];

    const uint me = sg * width + lane;     // this thread's rank in scan order
    const uint n = cols;
    const ulong base_row = ulong(row) * n;
    device const uint* keys_row = keys_in + base_row;
    device const uint* idx_row = idx_in + base_row;

    // (a) Count the row's digits.
    for (uint d = me; d < RADIX; d += bdim) {
        atomic_store_explicit(&hist[d], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = me; c < n; c += bdim) {
        atomic_fetch_add_explicit(&hist[digit_of(keys_row[c], shift)], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (b) Exclusive scan of the counts: where each bucket starts in the output row.
    uint total = 0;
    const uint count = me < RADIX ? atomic_load_explicit(&hist[me], memory_order_relaxed) : 0u;
    const uint start = block_exclusive_scan(count, group_sums, lane, sg, width, groups, total);
    if (me < RADIX) {
        offset[me] = start;
    }

    // (c) The tiles of the row, in order.
    const uint first = me * KEYS_PER_THREAD;
    for (uint base = 0; base < n; base += TILE) {
        const uint m = min(uint(TILE), n - base);

        for (uint j = me; j < m; j += bdim) {
            keys_s[j] = keys_row[base + j];
            idx_s[j] = idx_row[base + j];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint b = 0; b < RADIX_BITS; ++b) {
            const uint bit_pos = shift + b;
            uint k[KEYS_PER_THREAD];
            uint v[KEYS_PER_THREAD];
            uint ones = 0;
            for (uint i = 0; i < KEYS_PER_THREAD; ++i) {
                k[i] = 0u;
                v[i] = 0u;
                if (first + i < m) {
                    k[i] = keys_s[first + i];
                    v[i] = idx_s[first + i];
                    ones += (k[i] >> bit_pos) & 1u;
                }
            }
            uint ones_total = 0;
            uint ones_before = block_exclusive_scan(ones, group_sums, lane, sg, width, groups, ones_total);
            for (uint i = 0; i < KEYS_PER_THREAD; ++i) {
                const uint p = first + i;
                if (p < m) {
                    const uint bit = (k[i] >> bit_pos) & 1u;
                    const uint dst = bit == 0u ? p - ones_before : m - ones_total + ones_before;
                    keys_s[dst] = k[i];
                    idx_s[dst] = v[i];
                    ones_before += bit;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        for (uint p = me; p < m; p += bdim) {
            const uint d = digit_of(keys_s[p], shift);
            if (p == 0 || digit_of(keys_s[p - 1], shift) != d) {
                tile_start[d] = p;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = me; p < m; p += bdim) {
            const uint d = digit_of(keys_s[p], shift);
            const ulong dst = base_row + offset[d] + (p - tile_start[d]);
            keys_out[dst] = keys_s[p];
            idx_out[dst] = idx_s[p];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint p = me; p < m; p += bdim) {
            const uint d = digit_of(keys_s[p], shift);
            if (p == m - 1 || digit_of(keys_s[p + 1], shift) != d) {
                offset[d] += p + 1 - tile_start[d];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---------------------------------------------------------------- 3. gather
kernel void gather_rows(device const uint* probs       [[buffer(0)]],   // the float32 bits
                        device const uint* idx_sorted  [[buffer(1)]],
                        device uint*       sorted_bits [[buffer(2)]],   // sortedProbs, as bits
                        device int*        sorted_idx  [[buffer(3)]],
                        constant uint&     cols        [[buffer(4)]],
                        uint t    [[thread_position_in_threadgroup]],
                        uint row  [[threadgroup_position_in_grid]],
                        uint bdim [[threads_per_threadgroup]]) {
    const ulong base = ulong(row) * cols;
    for (uint p = t; p < cols; p += bdim) {
        const uint c = idx_sorted[base + p];
        sorted_idx[base + p] = int(c);
        sorted_bits[base + p] = probs[base + c];
    }
}
