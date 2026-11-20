// Lab 11.4 starter (Metal Shading Language). The kernels of kernel.cu:
// scan_segments and add_block_sums (Fig. 11.14), reduce_segments and scan_with_carry
// (Fig. 11.15), on top of the block scan of labs 11.1 and 11.2.
//
// CUDA                                  Metal
// #define COARSE_FACTOR 4 in kernel.cu  the macro COARSE_FACTOR, which the driver sets when
//                                       it builds the pipelines, from coarseFactor() in
//                                       launch.mm
// extern __shared__ float buffer_s[];   threadgroup float* buffer_s [[threadgroup(0)]],
//   + bytes in <<<grid, block, bytes>>>   sized by setThreadgroupMemoryLength (ch05-facts C8)
// __shared__ inside a __device__ helper not allowed: each kernel declares its threadgroup
//                                       arrays and passes pointers (ch11-facts C2)
// a null pointer for "no array"         separate kernels; Metal binds a buffer per index
//
// Threadgroups of one dispatch never talk to each other here: every hand-off between
// segments goes through a separate dispatch (launch.mm). launch.mm dispatches only
// blocks that are whole SIMD-groups, at most `width` of them, so stage 2 of the block
// scan and of the reduction fits in one SIMD-group.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSE_FACTOR
#error "COARSE_FACTOR is set by the lab driver when it builds the pipeline (from launch.mm)"
#endif

#define MAX_BLOCK_DIM 1024
#define MAX_SIMDGROUPS 32   // groups <= width and groups * width <= 1024

// ---------------------------------------------------------------- from labs 11.1 and 10.1
// Provided. Use yours instead if you like; keep the signatures.
inline float warp_scan(float val, uint lane, uint width) {
    for (uint stride = 1; stride < width; stride *= 2) {
        float leftVal = simd_shuffle_up(val, (ushort)stride);
        if (lane >= stride) {
            val += leftVal;
        }
    }
    return val;
}

inline float block_scan(float val, threadgroup float* warpSums_s,
                        uint lane, uint sg, uint width, uint groups) {
    val = warp_scan(val, lane, width);
    if (lane == width - 1) {
        warpSums_s[sg] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = lane < groups ? warpSums_s[lane] : 0.0f;
        s = warp_scan(s, lane, width);
        if (lane < groups) {
            warpSums_s[lane] = s;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg > 0) {
        val += warpSums_s[sg - 1];
    }
    return val;
}

// Lane 0 returns the sum of the SIMD-group's values (lab 10.1).
inline float warp_reduce(float val, uint width) {
    for (uint stride = width / 2; stride > 0; stride /= 2) {
        val += simd_shuffle_down(val, (ushort)stride);
    }
    return val;
}

// ---------------------------------------------------------------- the block-level scan
// Scans threadgroup b's segment of `input` into `output`, after adding `carry` to the
// segment's first element, and returns the segment's total in every thread.
inline float scan_segment(device const float* input, device float* output, float carry, uint N,
                          threadgroup float* buffer_s, threadgroup float* threadSums,
                          threadgroup float* warpSums_s,
                          uint t, uint b, uint bdim, uint lane, uint sg, uint width, uint groups) {
    // TODO 1: as in kernel.cu: your register_tiled_scan of lab 11.2, the carry into the
    //         first element, and the segment's total as the return value.
    return 0.0f;
}

// ---------------------------------------------------------------- Fig. 11.14
kernel void scan_segments(device const float* input     [[buffer(0)]],
                          device float*       output    [[buffer(1)]],
                          device float*       blockSums [[buffer(2)]],
                          constant uint&      N         [[buffer(3)]],
                          threadgroup float*  buffer_s  [[threadgroup(0)]],
                          uint t      [[thread_position_in_threadgroup]],
                          uint b      [[threadgroup_position_in_grid]],
                          uint bdim   [[threads_per_threadgroup]],
                          uint width  [[threads_per_simdgroup]],
                          uint lane   [[thread_index_in_simdgroup]],
                          uint sg     [[simdgroup_index_in_threadgroup]],
                          uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup float threadSums[MAX_BLOCK_DIM];
    threadgroup float warpSums_s[MAX_SIMDGROUPS];
    // TODO 2: scan the segment; one thread writes its total to blockSums[b].
}

kernel void add_block_sums(device float*       output           [[buffer(0)]],
                           device const float* scannedBlockSums [[buffer(1)]],
                           constant uint&      N                [[buffer(2)]],
                           uint t    [[thread_position_in_threadgroup]],
                           uint b    [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    // TODO 3: as AddBlockSumsKernel.
}

// ---------------------------------------------------------------- Fig. 11.15
kernel void reduce_segments(device const float* input     [[buffer(0)]],
                            device float*       blockSums [[buffer(1)]],
                            constant uint&      N         [[buffer(2)]],
                            uint t      [[thread_position_in_threadgroup]],
                            uint b      [[threadgroup_position_in_grid]],
                            uint bdim   [[threads_per_threadgroup]],
                            uint width  [[threads_per_simdgroup]],
                            uint lane   [[thread_index_in_simdgroup]],
                            uint sg     [[simdgroup_index_in_threadgroup]],
                            uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup float partialSums_s[MAX_SIMDGROUPS];
    // TODO 4: as ReduceSegmentsKernel, with warp_reduce.
}

kernel void scan_with_carry(device const float* input            [[buffer(0)]],
                            device float*       output           [[buffer(1)]],
                            device const float* scannedBlockSums [[buffer(2)]],
                            constant uint&      N                [[buffer(3)]],
                            threadgroup float*  buffer_s         [[threadgroup(0)]],
                            uint t      [[thread_position_in_threadgroup]],
                            uint b      [[threadgroup_position_in_grid]],
                            uint bdim   [[threads_per_threadgroup]],
                            uint width  [[threads_per_simdgroup]],
                            uint lane   [[thread_index_in_simdgroup]],
                            uint sg     [[simdgroup_index_in_threadgroup]],
                            uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup float threadSums[MAX_BLOCK_DIM];
    threadgroup float warpSums_s[MAX_SIMDGROUPS];
    // TODO 5: as ScanWithCarryKernel.
}
