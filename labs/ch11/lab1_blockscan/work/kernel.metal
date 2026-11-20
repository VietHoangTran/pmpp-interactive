// Lab 11.1 starter (Metal Shading Language). The kernels of kernel.cu:
// (a) ks_double_buffer; (b) warp_scan + block_scan_shuffle + warp_scan_shuffle; and,
// Metal only, (b2) block_scan_prefix + warp_scan_prefix, the same block scan with each
// SIMD-group scan done by one call to simd_prefix_inclusive_sum.
//
// CUDA                               Metal
// __shared__ float a[..]             threadgroup float a[..]   (in the kernel body only)
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// __shfl_up_sync(mask, v, d)         simd_shuffle_up(v, (ushort)d)    (no mask)
//
// Metal leaves the split of a threadgroup into SIMD-groups to the implementation, so
// take the SIMD-group index and the lane from the attributes above; do not compute them
// from the thread index. The SIMD-group width is not documented: read it, never assume
// it. A helper function cannot declare threadgroup memory: declare it in the kernel and
// pass a `threadgroup float*`. launch.mm dispatches only blocks that are whole
// SIMD-groups, at most `width` of them.
#include <metal_stdlib>
using namespace metal;

#define MAX_BLOCK_DIM 1024

// ---------------------------------------------------------------- (a)
kernel void ks_double_buffer(device const float* input  [[buffer(0)]],
                             device float*       output [[buffer(1)]],
                             constant uint&      N      [[buffer(2)]],
                             uint t    [[thread_position_in_threadgroup]],
                             uint b    [[threadgroup_position_in_grid]],
                             uint bdim [[threads_per_threadgroup]]) {
    // TODO 1: the steps of KoggeStoneDoubleBufferKernel in kernel.cu. A pointer into
    //         threadgroup memory is a `threadgroup float*`.
}

// ---------------------------------------------------------------- (b)
inline float warp_scan(float val, uint lane, uint width) {
    // TODO 3: the steps of warpScan in kernel.cu, with simd_shuffle_up.
    return val;
}

inline float block_scan_shuffle(float val, threadgroup float* warpSums_s,
                                uint lane, uint sg, uint width, uint groups) {
    // TODO 4: the steps of blockScan in kernel.cu, with warp_scan.
    return val;
}

kernel void warp_scan_shuffle(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              constant uint&      N      [[buffer(2)]],
                              uint t      [[thread_position_in_threadgroup]],
                              uint b      [[threadgroup_position_in_grid]],
                              uint bdim   [[threads_per_threadgroup]],
                              uint width  [[threads_per_simdgroup]],
                              uint lane   [[thread_index_in_simdgroup]],
                              uint sg     [[simdgroup_index_in_threadgroup]],
                              uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 5: the steps of WarpScanKernel in kernel.cu. Size the threadgroup array of
    //         SIMD-group totals from what launch.mm allows: groups <= width and
    //         groups * width = block <= 1024. What is the most groups can be?
}

// ---------------------------------------------------------------- (b2)
inline float block_scan_prefix(float val, threadgroup float* warpSums_s,
                               uint lane, uint sg, uint width, uint groups) {
    // TODO 7: block_scan_shuffle again, with each SIMD-group scan replaced by one call
    //         to simd_prefix_inclusive_sum. It sums over the ACTIVE threads of the
    //         SIMD-group (ch11 facts C1): what does that mean for a thread past N?
    return val;
}

kernel void warp_scan_prefix(device const float* input  [[buffer(0)]],
                             device float*       output [[buffer(1)]],
                             constant uint&      N      [[buffer(2)]],
                             uint t      [[thread_position_in_threadgroup]],
                             uint b      [[threadgroup_position_in_grid]],
                             uint bdim   [[threads_per_threadgroup]],
                             uint width  [[threads_per_simdgroup]],
                             uint lane   [[thread_index_in_simdgroup]],
                             uint sg     [[simdgroup_index_in_threadgroup]],
                             uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 8: as TODO 5, with block_scan_prefix.
}
