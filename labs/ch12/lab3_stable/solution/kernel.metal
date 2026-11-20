// Lab 12.3 reference solution (Metal Shading Language). The kernels of kernel.cu:
// keep_flags and scatter_flags (part A), count_kept and staged_scatter (part B). launch.mm
// encodes them, with the provided scan (../../shared-g12/scan_u32.metal) between them.
//
// CUDA                                   Metal
// __ballot_sync + __popc(mask)           simd_sum(keep ? 1u : 0u)        the SIMD-group's count
// __popc(mask & lanesBelow)              simd_prefix_exclusive_sum(...)  the rank among kept lanes
// extern __shared__ unsigned int s[]     threadgroup uint* s [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                        threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x % warpSize, / warpSize     [[thread_index_in_simdgroup]], [[simdgroup_index_in_threadgroup]]
// gridDim.x                              [[threadgroups_per_grid]]
//
// Notes:
//   - The SIMD-group functions act on the active threads (notes/today/ch12-facts.md C1). Every
//     thread calls them, kept or not, in uniform control flow, so "active" is the whole
//     SIMD-group and the question of which threads a branch leaves active does not arise (C3).
//   - simd_prefix_exclusive_sum and simd_sum need GPU family Apple7, which every Apple silicon
//     Mac is (infra-facts C2).
//   - The SIMD-group width is not a documented constant: launch.mm reads it from the pipeline
//     and sizes the per-SIMD-group threadgroup memory from it.
#include <metal_stdlib>
using namespace metal;

// ======================================================================== part A (exercise 1)

kernel void keep_flags(device const uint* input     [[buffer(0)]],
                       device uint*       keep      [[buffer(1)]],
                       constant uint&     n         [[buffer(2)]],
                       constant uint&     threshold [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    if (i < n) {
        keep[i] = input[i] < threshold ? 1u : 0u;
    }
}

kernel void scatter_flags(device const uint* input      [[buffer(0)]],
                          device const uint* keep       [[buffer(1)]],
                          device const uint* offset     [[buffer(2)]],
                          device uint*       output     [[buffer(3)]],
                          device uint*       outputSize [[buffer(4)]],
                          constant uint&     n          [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) {
        if (keep[i] != 0) {
            output[offset[i]] = input[i];
        }
        if (i == n - 1) {
            *outputSize = offset[i] + keep[i];
        }
    }
}

// ======================================================================== part B (exercises 3, 4)
// Threadgroup b owns keys [b * T * coarse, (b + 1) * T * coarse), T threads per threadgroup,
// walked in coarse rounds of T consecutive keys (coalesced).

kernel void count_kept(device const uint* input       [[buffer(0)]],
                       device uint*       blockCounts [[buffer(1)]],
                       constant uint&     n           [[buffer(2)]],
                       constant uint&     threshold   [[buffer(3)]],
                       constant uint&     coarse      [[buffer(4)]],
                       threadgroup uint*  groupTotals [[threadgroup(0)]],   // one per SIMD-group
                       uint t      [[thread_position_in_threadgroup]],
                       uint b      [[threadgroup_position_in_grid]],
                       uint T      [[threads_per_threadgroup]],
                       uint lane   [[thread_index_in_simdgroup]],
                       uint sg     [[simdgroup_index_in_threadgroup]],
                       uint groups [[simdgroups_per_threadgroup]]) {
    const uint tileStart = b * T * coarse;
    uint groupCount = 0;   // the same in every lane of a SIMD-group
    for (uint r = 0; r < coarse; ++r) {
        const uint i = tileStart + r * T + t;
        const bool keep = i < n && input[i] < threshold;
        groupCount += simd_sum(keep ? 1u : 0u);
    }
    if (lane == 0) {
        groupTotals[sg] = groupCount;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (t == 0) {
        uint total = 0;
        for (uint g = 0; g < groups; ++g) {
            total += groupTotals[g];
        }
        blockCounts[b] = total;
    }
}

kernel void staged_scatter(device const uint* input        [[buffer(0)]],
                           device uint*       output       [[buffer(1)]],
                           device const uint* blockOffsets [[buffer(2)]],
                           device uint*       outputSize   [[buffer(3)]],
                           constant uint&     n            [[buffer(4)]],
                           constant uint&     threshold    [[buffer(5)]],
                           constant uint&     coarse       [[buffer(6)]],
                           threadgroup uint*  staged_s     [[threadgroup(0)]],   // T * coarse keys
                           threadgroup uint*  groupTotals  [[threadgroup(1)]],   // one per SIMD-group
                           uint t        [[thread_position_in_threadgroup]],
                           uint b        [[threadgroup_position_in_grid]],
                           uint T        [[threads_per_threadgroup]],
                           uint numTiles [[threadgroups_per_grid]],
                           uint lane     [[thread_index_in_simdgroup]],
                           uint sg       [[simdgroup_index_in_threadgroup]],
                           uint groups   [[simdgroups_per_threadgroup]]) {
    const uint tileStart = b * T * coarse;
    uint staged = 0;   // kept keys staged so far; the same in every thread
    for (uint r = 0; r < coarse; ++r) {
        const uint i = tileStart + r * T + t;
        const uint key = i < n ? input[i] : 0u;
        const uint keep = (i < n && key < threshold) ? 1u : 0u;

        const uint rankInGroup = simd_prefix_exclusive_sum(keep);   // kept lanes below this one
        const uint groupTotal = simd_sum(keep);
        if (lane == 0) {
            groupTotals[sg] = groupTotal;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        uint groupsBefore = 0, roundTotal = 0;
        for (uint g = 0; g < groups; ++g) {
            const uint total = groupTotals[g];
            groupsBefore += g < sg ? total : 0u;
            roundTotal += total;
        }
        if (keep != 0) {
            staged_s[staged + groupsBefore + rankInGroup] = key;
        }
        staged += roundTotal;
        threadgroup_barrier(mem_flags::mem_threadgroup);   // groupTotals is rewritten next round
    }

    const uint base = blockOffsets[b];
    for (uint j = t; j < staged; j += T) {
        output[base + j] = staged_s[j];   // consecutive threads, consecutive keys
    }
    if (b == numTiles - 1 && t == 0) {
        *outputSize = base + staged;
    }
}
