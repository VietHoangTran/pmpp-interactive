// Lab 10.1 reference solution (Metal Shading Language). The same three kernels as
// kernel.cu: (a) shared_memory_sum, (b) warp_reduce + two_stage_warp_sum,
// (c) coarsened_sum.
//
// CUDA                               Metal
// __shared__ float a[..]             threadgroup float a[..]   (in the kernel body)
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// __shfl_down_sync(mask, v, d)       simd_shuffle_down(v, d)    (no mask)
// cuda::atomic_ref<float,...>        device atomic_float* and atomic_fetch_add_explicit
//
// Metal leaves the split of a threadgroup into SIMD-groups to the implementation, so the
// SIMD-group index and lane come from attributes, not from t / width
// (notes/today/ch04-facts.md C2). The SIMD-group width is not documented; launch.mm
// reads it from the pipeline and dispatches only blocks that are a multiple of it with
// at most `width` SIMD-groups, so stage 2 fits in one SIMD-group.
#include <metal_stdlib>
using namespace metal;

#define MAX_BLOCK_DIM 1024
// Per-SIMD-group sums. launch.mm guarantees groups <= width and groups = block / width
// <= 1024 / width, so groups <= 32 whatever the width is.
#define MAX_SIMDGROUPS 32

// ---------------------------------------------------------------- (a)
kernel void shared_memory_sum(device const float* input  [[buffer(0)]],
                              device float*       output [[buffer(1)]],
                              uint t    [[thread_position_in_threadgroup]],
                              uint b    [[threadgroup_position_in_grid]],
                              uint bdim [[threads_per_threadgroup]]) {
    threadgroup float input_s[MAX_BLOCK_DIM];
    device const float* row = input + 2 * bdim * b;
    input_s[t] = row[t] + row[t + bdim];
    for (uint stride = bdim / 2; stride >= 1; stride /= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (t == 0) {
        output[b] = input_s[0];
    }
}

// ---------------------------------------------------------------- (b)
// Lane 0 ends with the sum of the SIMD-group. The upper `stride` lanes get their own
// value back from simd_shuffle_down (infra-facts C2), as on CUDA.
inline float warp_reduce(float val, uint width) {
    float partialSum = val;
    for (uint stride = width / 2; stride > 0; stride /= 2) {
        partialSum += simd_shuffle_down(partialSum, (ushort)stride);
    }
    return partialSum;
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
    device const float* row = input + 2 * bdim * b;
    float partialSum = row[t] + row[t + bdim];
    partialSum = warp_reduce(partialSum, width);

    threadgroup float partialSums_s[MAX_SIMDGROUPS];
    if (lane == 0) {
        partialSums_s[sg] = partialSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (sg == 0) {
        float s = lane < groups ? partialSums_s[lane] : 0.0f;
        s = warp_reduce(s, width);
        if (lane == 0) {
            output[b] = s;
        }
    }
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
    uint segment = coarse * 2 * bdim * b;
    uint i = segment + t;
    float partialSum = i < N ? input[i] : 0.0f;
    for (uint c = 1; c < coarse * 2; ++c) {
        uint idx = i + c * bdim;
        if (idx < N) {
            partialSum += input[idx];
        }
    }

    partialSum = warp_reduce(partialSum, width);
    threadgroup float partialSums_s[MAX_SIMDGROUPS];
    if (lane == 0) {
        partialSums_s[sg] = partialSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = lane < groups ? partialSums_s[lane] : 0.0f;
        s = warp_reduce(s, width);
        if (lane == 0) {
            // Float atomics: device memory, Metal 3, Apple7 and later (infra-facts C1).
            atomic_fetch_add_explicit(output, s, memory_order_relaxed);
        }
    }
}
