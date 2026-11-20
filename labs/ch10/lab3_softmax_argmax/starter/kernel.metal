// Lab 10.3 starter (Metal Shading Language). The same three kernels as kernel.cu:
// row_max, softmax_rows, argmax_rows. One threadgroup of 256 threads per row; launch.mm
// dispatches them. The contract is the one at the top of kernel.cu.
//
// CUDA                               Metal
// __shared__ float s[MAX_WARPS]      threadgroup float* scratch [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// __shfl_down_sync(mask, v, d)       simd_shuffle_down(v, d)   (no mask)
// expf                               exp
//
// Metal also has whole-SIMD-group reductions that CUDA lacks for float: simd_max(v) and
// simd_sum(v) return the result to every active lane. Use them where they fit.
// The max identity is -FLT_MAX, not -INFINITY: see README.md.
#include <metal_stdlib>
using namespace metal;

// TODO 1: the block-level max and sum (two stages, as in kernel.cu). Each SIMD-group
//         reduces its values, lane 0 of each writes its result to scratch, then one
//         SIMD-group reduces those results. Lanes beyond `groups` hold the identity.
//         Every thread must end up with the result.

kernel void row_max(device const float* x   [[buffer(0)]],
                    device float*       out [[buffer(1)]],
                    constant uint&      cols [[buffer(2)]],
                    threadgroup float*  scratch [[threadgroup(0)]],
                    uint t      [[thread_position_in_threadgroup]],
                    uint row    [[threadgroup_position_in_grid]],
                    uint bdim   [[threads_per_threadgroup]],
                    uint lane   [[thread_index_in_simdgroup]],
                    uint sg     [[simdgroup_index_in_threadgroup]],
                    uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 2: threadgroup `row` reduces row `row`, with a loop strided by bdim.
}

kernel void softmax_rows(device const float* x     [[buffer(0)]],
                         device float*       y     [[buffer(1)]],
                         constant uint&      cols  [[buffer(2)]],
                         constant float&     scale [[buffer(3)]],
                         threadgroup float*  scratch [[threadgroup(0)]],
                         uint t      [[thread_position_in_threadgroup]],
                         uint row    [[threadgroup_position_in_grid]],
                         uint bdim   [[threads_per_threadgroup]],
                         uint lane   [[thread_index_in_simdgroup]],
                         uint sg     [[simdgroup_index_in_threadgroup]],
                         uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 3: the three passes of kernel.cu's TODO 5. The same question about the shared
    //         scratch applies here.
}

kernel void argmax_rows(device const float* x    [[buffer(0)]],
                        device int*         idx  [[buffer(1)]],
                        constant uint&      cols [[buffer(2)]],
                        threadgroup float*  vals [[threadgroup(0)]],
                        threadgroup int*    idxs [[threadgroup(1)]],
                        uint t      [[thread_position_in_threadgroup]],
                        uint row    [[threadgroup_position_in_grid]],
                        uint bdim   [[threads_per_threadgroup]],
                        uint width  [[threads_per_simdgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint sg     [[simdgroup_index_in_threadgroup]],
                        uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 4: (value, index) pairs with the combine of kernel.cu's TODO 7, ties to the
    //         lowest index. vals and idxs hold one pair per SIMD-group.
}
