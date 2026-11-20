// Lab X1.1: top-k by radix select (Metal Shading Language). The same two kernels as kernel.cu:
// topk_threshold and topk_gather, one threadgroup of BLOCK_SIZE threads per row; launch.mm
// encodes both. The contract is the one in README.md and at the top of kernel.cu.
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
// Things Metal makes you decide (README.md, "Portability"):
//   - simd_ballot's vote is 64 bits whatever the SIMD-group width, and its bits above the width
//     are undefined (ch12-facts C2). A rank popcount(vote & lanes below) never reads them; a
//     count or a first set lane must not trust them (lanes_present below helps).
//   - How a threadgroup splits into SIMD-groups is the implementation's (ch04-facts C2). If your
//     ranks run in (SIMD-group, lane) order, give each thread its column from
//     sg * width + lane, as lab 14.3 does, not from thread_position_in_threadgroup.
//   - The histogram must be atomic_uint; read it with atomic_load_explicit (ch09-facts C1).
#include <metal_stdlib>
using namespace metal;

#define BLOCK_SIZE 256                // threads per threadgroup; launch.mm's kBlock must be equal
#define DIGIT_BITS 8
#define BINS (1u << DIGIT_BITS)
#define SCRATCH_PER_ROW 4

// Provided: lab 14.3's key, ascending; -0.0 has the same key as +0.0.
inline uint ordered_key(uint u) {
    if (u == 0x80000000u) {
        u = 0u;
    }
    return (u & 0x80000000u) != 0u ? ~u : (u | 0x80000000u);
}

// Provided: the lanes that exist in a SIMD-group of `width` lanes, as bits of a 64-bit vote.
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
    // TODO: kernel.cu's TODOs 1 to 5, in MSL. The histogram is
    //       threadgroup atomic_uint hist[BINS];
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
    // TODO: kernel.cu's TODOs 6 to 8, in MSL. Per-SIMD-group totals can live in
    //       threadgroup arrays of BLOCK_SIZE entries, enough for any width.
}
