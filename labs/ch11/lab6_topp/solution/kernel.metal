// Lab 11.6 reference solution (Metal Shading Language): top_p_mask, the same kernel as
// kernel.cu, with Metal's one-call SIMD-group scans and sums in place of the shuffle loops.
// The contract is the one at the top of kernel.cu.
//
// Where Metal differs from the CUDA file:
//   - simd_prefix_exclusive_sum(x) gives each lane the sum of the lanes below it, and
//     simd_prefix_inclusive_sum(x) includes its own (ch11-facts C1). Apple does not document
//     the order of their additions, so README.md's error bound allows any order within a
//     SIMD-group.
//   - simd_sum(x) returns the SIMD-group's sum to every lane, for ints as for floats
//     (infra-facts C2).
//   - Lane and SIMD-group indices come from their attributes (ch11-facts C2); every position
//     below is built from them, never from the position in the threadgroup.
//   - The per-SIMD-group arrays have one slot per thread, enough for any width, because no
//     narrowest width is documented; launch.mm checks that one SIMD-group has a lane for
//     every SIMD-group.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_DIM 256        // threads per threadgroup (kBlock in launch.mm); one threadgroup per row
#define COARSE_FACTOR 4      // elements per thread per chunk

kernel void top_p_mask(device const float* probs      [[buffer(0)]],
                       device float*       massBefore [[buffer(1)]],
                       device uchar*       keep       [[buffer(2)]],
                       device int*         kept       [[buffer(3)]],
                       constant uint&      cols       [[buffer(4)]],
                       constant float&     p          [[buffer(5)]],
                       uint row         [[threadgroup_position_in_grid]],
                       uint lane        [[thread_index_in_simdgroup]],
                       uint simdgroup   [[simdgroup_index_in_threadgroup]],
                       uint simd_size   [[threads_per_simdgroup]],
                       uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float stage_s[COARSE_FACTOR * BLOCK_DIM];   // each SIMD-group's slice of the chunk
    threadgroup float simdSums_s[BLOCK_DIM];
    threadgroup int simdCounts_s[BLOCK_DIM];

    const ulong rowStart = (ulong)row * cols;
    device const float* rowProbs = probs + rowStart;
    device float* rowMass = massBefore + rowStart;
    device uchar* rowKeep = keep + rowStart;

    const uint chunkSize = COARSE_FACTOR * simd_size * simd_groups;
    const uint groupSlice = simdgroup * COARSE_FACTOR * simd_size;
    threadgroup float* myStage = stage_s + groupSlice;

    const bool keepAll = p >= 1.0f;   // llama.cpp's early return for p >= 1 (ch11-facts D2)
    int count = 0;
    float carry = 0.0f;

    for (uint chunk = 0; chunk < cols; chunk += chunkSize) {
        // Coalesced load of this SIMD-group's slice into threadgroup memory.
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            uint j = chunk + groupSlice + c * simd_size + lane;
            myStage[c * simd_size + lane] = (j < cols) ? rowProbs[j] : 0.0f;
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // Exclusive sequential scan of this lane's run.
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            run[c] = threadSum;
            threadSum += myStage[lane * COARSE_FACTOR + c];
        }

        // Block scan of the run totals, with one call per SIMD-group level.
        float exclusive = simd_prefix_exclusive_sum(threadSum);
        if (lane == simd_size - 1) {
            simdSums_s[simdgroup] = exclusive + threadSum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (simdgroup == 0) {
            float groupSum = (lane < simd_groups) ? simdSums_s[lane] : 0.0f;
            groupSum = simd_prefix_inclusive_sum(groupSum);
            if (lane < simd_groups) {
                simdSums_s[lane] = groupSum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float before = carry + ((simdgroup > 0) ? simdSums_s[simdgroup - 1] : 0.0f) + exclusive;
        carry += simdSums_s[simd_groups - 1];

        // Fused mass, keep decision and count.
        const uint first = chunk + groupSlice + lane * COARSE_FACTOR;
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            float mass = before + run[c];
            myStage[lane * COARSE_FACTOR + c] = mass;
            uint j = first + c;
            if (j < cols) {
                int keepIt = (j == 0 || keepAll || mass < p) ? 1 : 0;
                rowKeep[j] = uchar(keepIt);
                count += keepIt;
            }
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            uint j = chunk + groupSlice + c * simd_size + lane;
            if (j < cols) {
                rowMass[j] = myStage[c * simd_size + lane];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);   // simdSums_s is reused by the next chunk
    }

    // kept[row]: every SIMD-group sums its counts, then SIMD-group 0 sums those.
    int groupCount = simd_sum(count);
    if (lane == 0) {
        simdCounts_s[simdgroup] = groupCount;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simdgroup == 0) {
        int total = (lane < simd_groups) ? simdCounts_s[lane] : 0;
        total = simd_sum(total);
        if (lane == 0) {
            kept[row] = total;
        }
    }
}
