// Lab X1.2 starter (Metal Shading Language): min_p_count and filter_sorted_fused, the Metal
// counterparts of minPCount and filterSortedFused in kernel.cu. The contract is the one at the
// top of kernel.cu. The sort, softmax and top-p mask of part A are labs 14.3, 10.3 and 11.6's
// own kernel.metal files, which main.cpp loads from their directories.
//
// Metal's tools for this (README.md, "Portability"):
//   - simd_sum(x) gives every lane the SIMD-group's sum, for ints as for floats;
//     simd_prefix_exclusive_sum(x) and simd_prefix_inclusive_sum(x) scan in one call.
//   - Lane and SIMD-group indices come from their attributes. Build every position in a chunk
//     from them, not from the thread's position in the threadgroup.
//   - No narrowest SIMD-group width is documented: give per-SIMD-group threadgroup arrays one
//     slot per thread. launch.mm checks that one SIMD-group has a lane for every SIMD-group.
//   - Declare threadgroup arrays inside the kernel.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_DIM 256        // threads per threadgroup (kBlock in launch.mm); one threadgroup per row
#define COARSE_FACTOR 4      // tokens per thread per chunk

// ---------------------------------------------------------------- part A: min-p, then n
kernel void min_p_count(device const float* sortedLogits [[buffer(0)]],
                        device const int*   keptTopP     [[buffer(1)]],
                        device int*         n            [[buffer(2)]],
                        constant uint&      K            [[buffer(3)]],
                        constant float&     minP         [[buffer(4)]],
                        constant float&     lnMinP       [[buffer(5)]],
                        uint t           [[thread_position_in_threadgroup]],
                        uint row         [[threadgroup_position_in_grid]],
                        uint bdim        [[threads_per_threadgroup]],
                        uint lane        [[thread_index_in_simdgroup]],
                        uint simdgroup   [[simdgroup_index_in_threadgroup]],
                        uint simd_groups [[simdgroups_per_threadgroup]]) {
    // TODO 1: as kernel.cu's TODO 1: count the row's logits that min-p keeps, add the counts
    //         up across the threadgroup, and let one thread write n[row].
}

// ---------------------------------------------------------------- part B: fused
kernel void filter_sorted_fused(device const float* sortedLogits [[buffer(0)]],
                                device float*       sortedProbs  [[buffer(1)]],
                                device int*         n            [[buffer(2)]],
                                constant uint&      K            [[buffer(3)]],
                                constant float&     topP         [[buffer(4)]],
                                constant float&     minP         [[buffer(5)]],
                                constant float&     lnMinP       [[buffer(6)]],
                                uint t           [[thread_position_in_threadgroup]],
                                uint row         [[threadgroup_position_in_grid]],
                                uint bdim        [[threads_per_threadgroup]],
                                uint lane        [[thread_index_in_simdgroup]],
                                uint simdgroup   [[simdgroup_index_in_threadgroup]],
                                uint simd_size   [[threads_per_simdgroup]],
                                uint simd_groups [[simdgroups_per_threadgroup]]) {
    // TODO 2: as kernel.cu's TODO 4: pass 1 for the denominator, pass 2 in chunks of
    //         COARSE_FACTOR * BLOCK_DIM tokens with a carried prefix, both decisions while
    //         their values are in registers, then n[row]. Every threadgroup array you reuse
    //         needs a barrier between its last read and its next write.
}
