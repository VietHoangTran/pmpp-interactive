// Lab 12.2 starter (Metal): the same three unstable filters as kernel.cu, with the same THREE
// defects, keep = key < threshold. The caller zeroes outputSize. The harness defines
// BLOCK_DIM when it compiles this file; a threadgroup has at most that many threads.
//
// Not defects: the SIMD-group functions called from every thread, outside any branch, and
// the count > 0 test before the one atomic per SIMD-group (lab 12.1's README says why).
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

kernel void filter_atomic(device const uint* input      [[buffer(0)]],
                          device uint*       output     [[buffer(1)]],
                          constant uint&     n          [[buffer(2)]],
                          constant uint&     threshold  [[buffer(3)]],
                          device uint*       outputSize [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) {
        uint val = input[i];
        if (val < threshold) {
            uint j = *outputSize;                  // the next free slot
            *outputSize = j + 1;
            output[j] = val;
        }
    }
}

kernel void filter_warp_aggregated(device const uint*  input      [[buffer(0)]],
                                   device uint*        output     [[buffer(1)]],
                                   constant uint&      n          [[buffer(2)]],
                                   constant uint&      threshold  [[buffer(3)]],
                                   device atomic_uint* outputSize [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
    uint val = 0;
    bool keep = false;
    if (i < n) {
        val = input[i];
        keep = val < threshold;
    }

    uint flag = keep ? 1u : 0u;
    uint rank = simd_prefix_inclusive_sum(flag);
    uint count = simd_sum(flag);

    uint j = 0;
    if (simd_is_first() && count > 0) {
        j = atomic_fetch_add_explicit(outputSize, count, memory_order_relaxed);
    }
    j = simd_broadcast_first(j);

    if (keep) {
        output[j + rank] = val;
    }
}

kernel void filter_privatised(device const uint*  input      [[buffer(0)]],
                              device uint*        output     [[buffer(1)]],
                              constant uint&      n          [[buffer(2)]],
                              constant uint&      threshold  [[buffer(3)]],
                              device atomic_uint* outputSize [[buffer(4)]],
                              uint i   [[thread_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
    threadgroup uint output_s[BLOCK_DIM];
    threadgroup atomic_uint outputSize_s;
    if (tid == 0) {
        atomic_store_explicit(&outputSize_s, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (i < n) {
        uint val = input[i];
        if (val < threshold) {
            uint j = atomic_fetch_add_explicit(&outputSize_s, 1u, memory_order_relaxed);
            output_s[j] = val;
        }
    }

    threadgroup uint j_s;
    if (tid == 0) {
        uint count = atomic_load_explicit(&outputSize_s, memory_order_relaxed);
        j_s = atomic_fetch_add_explicit(outputSize, count, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    uint count = atomic_load_explicit(&outputSize_s, memory_order_relaxed);
    if (tid < count) {
        output[j_s + tid] = output_s[tid];
    }
}
