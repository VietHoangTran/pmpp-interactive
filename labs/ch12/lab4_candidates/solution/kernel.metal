// Lab 12.4 reference solution (Metal Shading Language). The same kernel as kernel.cu:
// compact_candidates. One threadgroup of 256 threads per row; launch.mm dispatches it.
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
// Notes (notes/today/ch12-facts.md):
//   - The vote is 64 bits whatever the SIMD-group width, and on hardware with fewer than 64
//     lanes its top bits are undefined (C2). The rank, popcount(vote & lanes below), never
//     touches them, because the lane index is below the width. The SIMD-group's total would:
//     it comes from simd_sum instead (C1, C2).
//   - Metal also ranks in one call: simd_prefix_exclusive_sum(keep ? 1u : 0u) (C1). This
//     file uses the vote so that it reads like kernel.cu.
//   - Every SIMD-group function is called in uniform control flow, so the question of which
//     threads are active inside a branch, which Apple does not answer (C3), does not arise.
//   - One threadgroup per row and no waiting between threadgroups: the design C4 calls
//     documented-safe on Metal, and the one llama.cpp's Metal top-k uses (D2).
//   - No arithmetic touches a value. Metal's math modes relax arithmetic (reassociation,
//     contraction, reciprocals) and the handling of NaN, INF and signed zeros (ch05-facts
//     C7); a comparison of two finite floats and a copy involve none of those (our reading),
//     so outVal is the loaded value bit for bit and the test compares exactly.
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
    const ulong rowStart = ulong(row) * cols;        // threadgroup r owns row r
    device const float* in = logits + rowStart;
    device int* idxOut = outIdx + rowStart;
    device float* valOut = outVal + rowStart;
    const float thr = threshold[row];

    const simd_vote::vote_t lanesBelow = (simd_vote::vote_t(1) << lane) - 1;

    uint rowCount = 0;                               // the same in every thread
    for (uint base = 0; base < cols; base += bdim) {
        const uint c = base + t;
        float val = 0.0f;
        bool keep = false;
        if (c < cols) {
            val = in[c];
            keep = val >= thr;
        }

        const simd_vote::vote_t vote = simd_vote::vote_t(simd_ballot(keep));
        const uint rankInGroup = uint(popcount(vote & lanesBelow));
        const uint groupTotal = simd_sum(keep ? 1u : 0u);
        if (lane == 0) {
            groupTotals[sg] = groupTotal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint before = 0, roundTotal = 0;
        for (uint g = 0; g < groups; ++g) {
            const uint n = groupTotals[g];
            before += g < sg ? n : 0;
            roundTotal += n;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (keep) {
            const uint k = rowCount + before + rankInGroup;
            idxOut[k] = int(c);
            valOut[k] = val;
        }
        rowCount += roundTotal;
    }

    if (t == 0) {
        counts[row] = rowCount;
    }
}
