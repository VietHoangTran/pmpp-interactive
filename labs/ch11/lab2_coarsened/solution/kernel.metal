// Lab 11.2 reference solution (Metal Shading Language). The kernels of kernel.cu:
// (a) coarsened_scan, (b) register_tiled_scan, and the block scan of lab 11.1.
//
// CUDA                                  Metal
// template <unsigned COARSE_FACTOR>     the macro COARSE_FACTOR, set by the driver when it
//                                       builds the pipeline (one pipeline per factor)
// extern __shared__ float buffer_s[];   threadgroup float* buffer_s [[threadgroup(0)]],
//   + bytes in <<<grid, block, bytes>>>   sized by setThreadgroupMemoryLength (ch05-facts C8)
// #pragma unroll                        none: MSL documents no unroll control (infra-facts,
//                                       "Where the backends really differ"); the loops
//                                       have constant trip counts and the compiler decides
//
// launch.mm dispatches only blocks that are whole SIMD-groups, at most `width` of them,
// and only when the threadgroup memory fits the device.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSE_FACTOR
#error "COARSE_FACTOR is set by the lab driver when it builds the pipeline"
#endif

#define MAX_BLOCK_DIM 1024
#define MAX_SIMDGROUPS 32   // groups <= width and groups * width <= 1024

// ---------------------------------------------------------------- from lab 11.1
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

// ---------------------------------------------------------------- (a)
kernel void coarsened_scan(device const float* input    [[buffer(0)]],
                           device float*       output   [[buffer(1)]],
                           constant uint&      N        [[buffer(2)]],
                           threadgroup float*  buffer_s [[threadgroup(0)]],
                           uint t      [[thread_position_in_threadgroup]],
                           uint b      [[threadgroup_position_in_grid]],
                           uint bdim   [[threads_per_threadgroup]],
                           uint width  [[threads_per_simdgroup]],
                           uint lane   [[thread_index_in_simdgroup]],
                           uint sg     [[simdgroup_index_in_threadgroup]],
                           uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup float threadSums[MAX_BLOCK_DIM];
    threadgroup float warpSums_s[MAX_SIMDGROUPS];
    uint blockSegment = b * COARSE_FACTOR * bdim;

    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        uint idx = blockSegment + c * bdim + t;
        buffer_s[c * bdim + t] = idx < N ? input[idx] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint threadSegment = t * COARSE_FACTOR;
    for (uint c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];
    }

    float threadSum = buffer_s[threadSegment + COARSE_FACTOR - 1];
    threadSum = block_scan(threadSum, warpSums_s, lane, sg, width, groups);
    threadSums[t] = threadSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (t > 0) {
        float prevThreadSum = threadSums[t - 1];
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[threadSegment + c] += prevThreadSum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        uint idx = blockSegment + c * bdim + t;
        if (idx < N) {
            output[idx] = buffer_s[c * bdim + t];
        }
    }
}

// ---------------------------------------------------------------- (b)
kernel void register_tiled_scan(device const float* input    [[buffer(0)]],
                                device float*       output   [[buffer(1)]],
                                constant uint&      N        [[buffer(2)]],
                                threadgroup float*  buffer_s [[threadgroup(0)]],
                                uint t      [[thread_position_in_threadgroup]],
                                uint b      [[threadgroup_position_in_grid]],
                                uint bdim   [[threads_per_threadgroup]],
                                uint width  [[threads_per_simdgroup]],
                                uint lane   [[thread_index_in_simdgroup]],
                                uint sg     [[simdgroup_index_in_threadgroup]],
                                uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup float threadSums[MAX_BLOCK_DIM];
    threadgroup float warpSums_s[MAX_SIMDGROUPS];
    uint blockSegment = b * COARSE_FACTOR * bdim;

    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        uint idx = blockSegment + c * bdim + t;
        buffer_s[c * bdim + t] = idx < N ? input[idx] : 0.0f;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint threadSegment = t * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
    for (uint c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];
    }

    float threadSum = block_scan(buffer_r[COARSE_FACTOR - 1], warpSums_s, lane, sg, width, groups);
    threadSums[t] = threadSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float prevPartialSum = t > 0 ? threadSums[t - 1] : 0.0f;
    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prevPartialSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        uint idx = blockSegment + c * bdim + t;
        if (idx < N) {
            output[idx] = buffer_s[c * bdim + t];
        }
    }
}
