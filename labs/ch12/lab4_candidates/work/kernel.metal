// Lab 12.4 starter (Metal Shading Language). The same kernel as kernel.cu:
// compact_candidates. One threadgroup of 256 threads per row; launch.mm dispatches it. The
// contract is the one at the top of kernel.cu.
//
// CUDA                                  Metal
// __shared__ unsigned int s[MAX_WARPS]  threadgroup uint* groupTotals [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                       threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                              [[threads_per_simdgroup]]
// threadIdx.x % warpSize                [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize                [[simdgroup_index_in_threadgroup]]
// __ballot_sync(full, keep)             simd_vote::vote_t(simd_ballot(keep))   (no mask; 64 bits)
// __popc / __popcll                     popcount
//
// Two facts about the vote (README: Portability): it is 64 bits whatever the SIMD-group
// width, and when the SIMD-group has fewer than 64 lanes its top bits are undefined. Metal
// also has simd_sum and simd_prefix_exclusive_sum, which count over a SIMD-group in one call.
#include <metal_stdlib>
using namespace metal;

kernel void compact_candidates(device const float* logits      [[buffer(0)]],
                               device const float* threshold   [[buffer(1)]],
                               device int*         outIdx      [[buffer(2)]],
                               device float*       outVal      [[buffer(3)]],
                               device uint*        counts      [[buffer(4)]],
                               constant uint&      cols        [[buffer(5)]],
                               threadgroup uint*   groupTotals [[threadgroup(0)]],
                               uint t      [[thread_position_in_threadgroup]],
                               uint row    [[threadgroup_position_in_grid]],
                               uint bdim   [[threads_per_threadgroup]],
                               uint lane   [[thread_index_in_simdgroup]],
                               uint sg     [[simdgroup_index_in_threadgroup]],
                               uint groups [[simdgroups_per_threadgroup]]) {
    // TODO: kernel.cu's TODOs 2 to 4, with a SIMD-group for a warp. Call the SIMD-group
    //       functions where every thread of the threadgroup calls them (not inside an `if`
    //       that some threads skip), and count a SIMD-group's survivors without reading
    //       the vote's undefined top bits.
}
