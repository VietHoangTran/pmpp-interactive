// Lab 11.1 reference solution (Metal Shading Language). The kernels of kernel.cu:
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
// Metal leaves the split of a threadgroup into SIMD-groups to the implementation, so the
// SIMD-group index and the lane come from attributes, not from t / width (ch04-facts C2).
// The width is not documented (infra-facts C5): read it. A helper cannot declare
// threadgroup memory (ch11-facts C2), so each kernel declares the SIMD-group totals and
// passes a pointer. launch.mm dispatches only blocks that are whole SIMD-groups with at
// most `width` of them, so stage 2 fits in one SIMD-group.
#include <metal_stdlib>
using namespace metal;

#define MAX_BLOCK_DIM 1024
// SIMD-group totals. launch.mm guarantees groups <= width and groups * width = block
// <= 1024, so groups <= 32 whatever the width is.
#define MAX_SIMDGROUPS 32

// ---------------------------------------------------------------- (a)
kernel void ks_double_buffer(device const float* input  [[buffer(0)]],
                             device float*       output [[buffer(1)]],
                             constant uint&      N      [[buffer(2)]],
                             uint t    [[thread_position_in_threadgroup]],
                             uint b    [[threadgroup_position_in_grid]],
                             uint bdim [[threads_per_threadgroup]]) {
    threadgroup float buffer1_s[MAX_BLOCK_DIM];
    threadgroup float buffer2_s[MAX_BLOCK_DIM];
    uint i = b * bdim + t;
    threadgroup float* inBuffer_s = buffer1_s;
    threadgroup float* outBuffer_s = buffer2_s;
    inBuffer_s[t] = i < N ? input[i] : 0.0f;
    for (uint stride = 1; stride < bdim; stride *= 2) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (t >= stride) {
            outBuffer_s[t] = inBuffer_s[t] + inBuffer_s[t - stride];
        } else {
            outBuffer_s[t] = inBuffer_s[t];
        }
        threadgroup float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if (i < N) {
        output[i] = inBuffer_s[t];
    }
}

// ---------------------------------------------------------------- (b)
// A lane below `stride` gets its own value back from simd_shuffle_up (ch11-facts C1).
inline float warp_scan(float val, uint lane, uint width) {
    for (uint stride = 1; stride < width; stride *= 2) {
        float leftVal = simd_shuffle_up(val, (ushort)stride);
        if (lane >= stride) {
            val += leftVal;
        }
    }
    return val;
}

inline float block_scan_shuffle(float val, threadgroup float* warpSums_s,
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
    threadgroup float warpSums_s[MAX_SIMDGROUPS];
    uint i = b * bdim + t;
    float val = i < N ? input[i] : 0.0f;
    val = block_scan_shuffle(val, warpSums_s, lane, sg, width, groups);
    if (i < N) {
        output[i] = val;
    }
}

// ---------------------------------------------------------------- (b2)
// The same three stages; each SIMD-group scan is one call. The functions act on the
// active threads only (ch11-facts C1), which is why out-of-range threads stay in the
// kernel and feed 0. Their summation order is not documented.
inline float block_scan_prefix(float val, threadgroup float* warpSums_s,
                               uint lane, uint sg, uint width, uint groups) {
    val = simd_prefix_inclusive_sum(val);
    if (lane == width - 1) {
        warpSums_s[sg] = val;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        float s = lane < groups ? warpSums_s[lane] : 0.0f;
        s = simd_prefix_inclusive_sum(s);
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
    threadgroup float warpSums_s[MAX_SIMDGROUPS];
    uint i = b * bdim + t;
    float val = i < N ? input[i] : 0.0f;
    val = block_scan_prefix(val, warpSums_s, lane, sg, width, groups);
    if (i < N) {
        output[i] = val;
    }
}
