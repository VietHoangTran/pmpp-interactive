// Lab 14.3 starter (Metal Shading Language). The same three kernels as kernel.cu:
// desc_keys, sort_rows_pass, gather_rows. One threadgroup of BLOCK_SIZE threads per row;
// launch.mm encodes the whole sort. The contract is the one at the top of kernel.cu.
//
// CUDA                                 Metal
// __shared__ unsigned int a[N]         threadgroup uint a[N];   (declared inside the kernel)
// cuda::atomic_ref<...> on shared      threadgroup atomic_uint a[N]; with
//                                      atomic_store_explicit / atomic_fetch_add_explicit /
//                                      atomic_load_explicit and memory_order_relaxed
// __syncthreads()                      threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                             [[threads_per_simdgroup]]
// (const unsigned int*)probs           device const uint* probs: the host buffer holds float32
//
// probs and sortedProbs are bound as uint on purpose: do every step on the bits, never on a
// float (README.md, "Why bits, not floats": Metal's math modes may rewrite float tests).
// Metal leaves the split of a threadgroup into SIMD-groups to the implementation, and the
// provided scan counts in (SIMD-group, lane) order: rank your threads the same way.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256                          // threads per threadgroup; one per row
#define RADIX_BITS 8                            // r; launch.mm's kRadixBits must be equal
#define RADIX (1u << RADIX_BITS)
#define KEYS_PER_THREAD 4                       // c
#define TILE (BLOCK_SIZE * KEYS_PER_THREAD)

// ---------------------------------------------------------------- provided: block scan
// Chapter 11's scan-scan-add on unsigned counts, with Metal's SIMD-group prefix sum in place
// of the shuffle loop. Returns the sum of v over the threads ranked before this one, where
// the rank is sg * width + lane, and the threadgroup's total in `total`. Every thread must
// call it. group_sums has one entry per SIMD-group (launch.mm sizes it), and there must be no
// more SIMD-groups than lanes. Two calls in a row need a barrier between them.
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
    // TODO 1: kernel.cu's TODO 1.
}

// ---------------------------------------------------------------- 2. one digit
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
    // TODO 2: kernel.cu's TODOs 2 to 4. Declare the threadgroup arrays here; the histogram
    //         is atomic_uint. Rank the threads as the scan does, not by thread position.
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
    // TODO 3: kernel.cu's TODO 5.
}
