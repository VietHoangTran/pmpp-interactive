// Lab 12.3 starter (Metal Shading Language). The kernels of kernel.cu: keep_flags and
// scatter_flags (part A), count_kept and staged_scatter (part B). launch.mm encodes them, with
// the provided scan (../../shared-g12/scan_u32.metal) between them. The signatures are given;
// keep the buffer indices, launch.mm relies on them.
//
// CUDA                                   Metal
// __ballot_sync + __popc(mask)           simd_sum(keep ? 1u : 0u)        the SIMD-group's count
// __popc(mask & lanesBelow)              simd_prefix_exclusive_sum(...)  the rank among kept lanes
// extern __shared__ unsigned int s[]     threadgroup uint* s [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                        threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x % warpSize, / warpSize     [[thread_index_in_simdgroup]], [[simdgroup_index_in_threadgroup]]
// gridDim.x                              [[threadgroups_per_grid]]
//
// The SIMD-group functions act on the active threads (notes/today/ch12-facts.md C1). Call them
// from every thread, kept or not, so that "active" means the whole SIMD-group.
#include <metal_stdlib>
using namespace metal;

// ======================================================================== part A (exercise 1)

kernel void keep_flags(device const uint* input     [[buffer(0)]],
                       device uint*       keep      [[buffer(1)]],
                       constant uint&     n         [[buffer(2)]],
                       constant uint&     threshold [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    // TODO A1: as keepFlagsKernel.
}

kernel void scatter_flags(device const uint* input      [[buffer(0)]],
                          device const uint* keep       [[buffer(1)]],
                          device const uint* offset     [[buffer(2)]],
                          device uint*       output     [[buffer(3)]],
                          device uint*       outputSize [[buffer(4)]],
                          constant uint&     n          [[buffer(5)]],
                          uint i [[thread_position_in_grid]]) {
    // TODO A2, A3: as scatterKernel.
}

// ======================================================================== part B (exercises 3, 4)
// Threadgroup b owns keys [b * T * coarse, (b + 1) * T * coarse), T threads per threadgroup,
// walked in coarse rounds of T consecutive keys.

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
    // TODO B1, B2: as countKeptKernel, with a SIMD-group sum in place of vote and popcount.
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
    // TODO B4 to B6: as stagedScatterKernel. The rank among the kept lanes below this one is
    //                one SIMD-group function; so is the SIMD-group's total.
}
