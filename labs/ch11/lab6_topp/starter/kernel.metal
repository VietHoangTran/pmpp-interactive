// Lab 11.6 starter (Metal Shading Language): top_p_mask, the same kernel as kernel.cu. One
// threadgroup of BLOCK_DIM threads per row; launch.mm dispatches it. The contract is the one
// at the top of kernel.cu.
//
// CUDA                               Metal
// __shared__ float s[N];             threadgroup float s[N];  declared in the kernel, not a helper
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// __syncwarp()                       simdgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// laneIdx(), warpIdx()               [[thread_index_in_simdgroup]], [[simdgroup_index_in_threadgroup]]
// blockDim.x / warpSize              [[simdgroups_per_threadgroup]]
// unsigned char                      uchar
//
// Metal also does in one call what Fig. 11.8 does in a loop: simd_prefix_inclusive_sum(v) and
// simd_prefix_exclusive_sum(v) scan across the SIMD-group, and simd_sum(v) returns the
// SIMD-group's sum to every lane, for ints too (notes/today/ch11-facts.md C1). Use them where
// they fit. Build every position from the lane and SIMD-group indices, never from the thread's
// position in the threadgroup (ch11-facts C2).
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
    // TODO 1: threadgroup `row` handles row `row`: lab 11.3's chunk loop with its carried
    //         prefix, in Metal. Size the per-SIMD-group arrays for any width: Apple documents
    //         none, so one slot per thread is the safe bound.
    // TODO 2: fused, as in kernel.cu's TODO 2: the keep rule on each token's mass, keep
    //         written, kept tokens counted.
    // TODO 3: kept[row], from every thread's count.
}
