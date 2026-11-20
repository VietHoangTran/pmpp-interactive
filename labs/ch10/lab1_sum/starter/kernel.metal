// Lab 10.1 starter (Metal Shading Language). The same three kernels as kernel.cu:
// (a) shared_memory_sum, (b) warp_reduce + two_stage_warp_sum, (c) coarsened_sum.
//
// CUDA                               Metal
// __shared__ float a[..]             threadgroup float a[..]   (in the kernel body)
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// __shfl_down_sync(mask, v, d)       simd_shuffle_down(v, (ushort)d)   (no mask)
// cuda::atomic_ref<float,...>        device atomic_float* and
//   .fetch_add(v, relaxed)             atomic_fetch_add_explicit(p, v, memory_order_relaxed)
//
// Metal leaves the split of a threadgroup into SIMD-groups to the implementation, so
// take the SIMD-group index and the lane from the attributes above; do not compute them
// from the thread index. The SIMD-group width is not documented: read it, never assume
// it. launch.mm dispatches only blocks that are whole SIMD-groups, at most `width` of
// them.
#include <metal_stdlib>
using namespace metal;

#define MAX_BLOCK_DIM 1024

// ---------------------------------------------------------------- (a)
kernel void shared_memory_sum(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              uint t    [[thread_position_in_threadgroup]],
                              uint b    [[threadgroup_position_in_grid]],
                              uint bdim [[threads_per_threadgroup]]) {
    // TODO 1: the steps of SharedMemorySumReductionKernel in kernel.cu.
}

// ---------------------------------------------------------------- (b)
// Every lane of a SIMD-group calls this; lane 0 returns the SIMD-group's sum.
inline float warp_reduce(float val, uint width) {
    // TODO 3: shuffle down with strides width/2, ..., 2, 1, adding as you go.
    return val;
}

kernel void two_stage_warp_sum(device const float* input  [[buffer(0)]],
                               device float*       output [[buffer(1)]],
                               uint t      [[thread_position_in_threadgroup]],
                               uint b      [[threadgroup_position_in_grid]],
                               uint bdim   [[threads_per_threadgroup]],
                               uint width  [[threads_per_simdgroup]],
                               uint lane   [[thread_index_in_simdgroup]],
                               uint sg     [[simdgroup_index_in_threadgroup]],
                               uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 4: the steps of TwoStageWarpLevelSumReductionKernel in kernel.cu. Size the
    //         threadgroup array of per-SIMD-group sums from what launch.mm allows:
    //         groups <= width and groups * width = block <= 1024. What is the most
    //         groups can be?
}

// ---------------------------------------------------------------- (c)
kernel void coarsened_sum(device const float* input  [[buffer(0)]],
                          device atomic_float* output [[buffer(1)]],
                          constant uint& N           [[buffer(2)]],
                          constant uint& coarse      [[buffer(3)]],
                          uint t      [[thread_position_in_threadgroup]],
                          uint b      [[threadgroup_position_in_grid]],
                          uint bdim   [[threads_per_threadgroup]],
                          uint width  [[threads_per_simdgroup]],
                          uint lane   [[thread_index_in_simdgroup]],
                          uint sg     [[simdgroup_index_in_threadgroup]],
                          uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 6: the steps of CoarsenedSumReductionKernel in kernel.cu.
}
