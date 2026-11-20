// Provided for lab 11.2, not edited: the thread-major scan of bs_threadmajor.cu in Metal
// Shading Language. Each thread reads its own COARSE_FACTOR consecutive elements
// directly from device memory, with no staging through threadgroup memory.
//
// COARSE_FACTOR is a preprocessor macro the driver sets when it builds the pipeline.
// The SIMD-group width, lane and SIMD-group index come from attributes (infra-facts C5);
// bs_launch.mm dispatches only blocks that are whole SIMD-groups, at most `width` of them.
// MSL documents no unroll pragma ("Where the backends really differ" in infra-facts), so
// the loops have constant trip counts and the compiler decides.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSE_FACTOR
#error "COARSE_FACTOR is set by the lab driver when it builds the pipeline"
#endif

#define TM_MAX_BLOCK_DIM 1024
#define TM_MAX_SIMDGROUPS 32   // groups <= width and groups * width <= 1024

inline float tm_warp_scan(float val, uint lane, uint width) {
    for (uint stride = 1; stride < width; stride *= 2) {
        float leftVal = simd_shuffle_up(val, (ushort)stride);
        if (lane >= stride) {
            val += leftVal;
        }
    }
    return val;
}

inline float tm_block_scan(float val, threadgroup float* warpSums_s,
                           uint lane, uint sg, uint width, uint groups) {
    val = tm_warp_scan(val, lane, width);
    if (lane == width - 1) {
        warpSums_s[sg] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = lane < groups ? warpSums_s[lane] : 0.0f;
        s = tm_warp_scan(s, lane, width);
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

kernel void thread_major_scan(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              constant uint&      N      [[buffer(2)]],
                              uint t      [[thread_position_in_threadgroup]],
                              uint b      [[threadgroup_position_in_grid]],
                              uint bdim   [[threads_per_threadgroup]],
                              uint width  [[threads_per_simdgroup]],
                              uint lane   [[thread_index_in_simdgroup]],
                              uint sg     [[simdgroup_index_in_threadgroup]],
                              uint groups [[simdgroups_per_threadgroup]]) {
    threadgroup float threadSums[TM_MAX_BLOCK_DIM];
    threadgroup float warpSums_s[TM_MAX_SIMDGROUPS];

    uint threadSegment = b * COARSE_FACTOR * bdim + t * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        uint idx = threadSegment + c;
        float x = idx < N ? input[idx] : 0.0f;
        buffer_r[c] = c == 0 ? x : buffer_r[c - 1] + x;
    }

    float threadSum = tm_block_scan(buffer_r[COARSE_FACTOR - 1], warpSums_s, lane, sg, width, groups);
    threadSums[t] = threadSum;
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float prevPartialSum = t > 0 ? threadSums[t - 1] : 0.0f;
    for (uint c = 0; c < COARSE_FACTOR; ++c) {
        uint idx = threadSegment + c;
        if (idx < N) {
            output[idx] = buffer_r[c] + prevPartialSum;
        }
    }
}
