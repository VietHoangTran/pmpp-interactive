// Lab 14.2 starter (Metal Shading Language). The same two kernels as kernel.cu; launch.mm
// drives them.
//
// CUDA                                  Metal
// extern __shared__ unsigned int s[]    threadgroup uint* s [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                       threadgroup_barrier(mem_flags::mem_threadgroup)
// gridDim.x                             [[threadgroups_per_grid]]
//
// For the block-level scan of each split you may use simd_prefix_exclusive_sum and simd_sum
// inside each SIMD-group, called by every thread (notes/today/infra-facts.md C2), plus one
// threadgroup slot per SIMD-group for their totals: sgTotals below, which launch.mm sizes from
// the pipeline's threadExecutionWidth. The SIMD-group width is not a documented constant
// (infra C5): never write a number for it. If you scan that way, give each thread the tile
// positions of its rank in the scan's order, sg * width + lane, not of
// thread_position_in_threadgroup: how a threadgroup is split into SIMD-groups is
// implementation-defined (notes/today/ch04-facts.md C2).
#include <metal_stdlib>
using namespace metal;

#define BLOCK_DIM 256
#define MAX_KEYS_PER_THREAD 8
#define MAX_RADIX_BITS 8
#define MAX_BUCKETS 256   // 2^MAX_RADIX_BITS

kernel void radix_local_sort(device const uint* keysIn        [[buffer(0)]],
                             device const uint* valsIn        [[buffer(1)]],
                             device uint*       keysLocal     [[buffer(2)]],
                             device uint*       valsLocal     [[buffer(3)]],
                             device uint*       table         [[buffer(4)]],
                             device uint*       localStarts   [[buffer(5)]],
                             constant uint&     n             [[buffer(6)]],
                             constant uint&     shift         [[buffer(7)]],
                             constant uint&     radixBits     [[buffer(8)]],
                             constant uint&     keysPerThread [[buffer(9)]],
                             threadgroup uint*  tile_s        [[threadgroup(0)]],   // keys, then values: 2 * 256c
                             threadgroup uint*  sgTotals      [[threadgroup(1)]],   // one per SIMD-group
                             uint t         [[thread_position_in_threadgroup]],
                             uint block     [[threadgroup_position_in_grid]],
                             uint numBlocks [[threadgroups_per_grid]],
                             uint lane      [[thread_index_in_simdgroup]],
                             uint sg        [[simdgroup_index_in_threadgroup]],
                             uint numSg     [[simdgroups_per_threadgroup]],
                             uint width     [[threads_per_simdgroup]]) {
    // TODO: the five steps of radixLocalSort in kernel.cu.
}

kernel void radix_scatter_local(device const uint* keysLocal     [[buffer(0)]],
                                device const uint* valsLocal     [[buffer(1)]],
                                device uint*       keysOut       [[buffer(2)]],
                                device uint*       valsOut       [[buffer(3)]],
                                device const uint* tableScanned  [[buffer(4)]],
                                device const uint* localStarts   [[buffer(5)]],
                                constant uint&     n             [[buffer(6)]],
                                constant uint&     shift         [[buffer(7)]],
                                constant uint&     radixBits     [[buffer(8)]],
                                constant uint&     keysPerThread [[buffer(9)]],
                                uint t         [[thread_position_in_threadgroup]],
                                uint block     [[threadgroup_position_in_grid]],
                                uint numBlocks [[threadgroups_per_grid]]) {
    // TODO: as radixScatterLocal.
}
