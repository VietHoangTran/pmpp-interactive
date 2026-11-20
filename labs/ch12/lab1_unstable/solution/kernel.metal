// Lab 12.1 reference solution (Metal Shading Language): the same three unstable filters as
// kernel.cu (book Figs. 12.2, 12.3 and 12.6, pp. 291 to 296), keep = key < threshold.
// The caller zeroes outputSize. The harness defines BLOCK_DIM when it compiles this file.
//
// What changes in Metal:
//   - Atomic functions take atomic types only, so the counter is `device atomic_uint*` and
//     the private counter is a `threadgroup atomic_uint` (notes/today/infra-facts.md C1).
//     A read-modify-write takes no scope: the address space says who shares the counter.
//   - Metal has no lane masks to pass around and no cooperative groups. Each step of the
//     warp-level append is one SIMD-group function (notes/today/ch12-facts.md C1):
//       vote + count       simd_sum(keep ? 1 : 0)
//       rank among keepers simd_prefix_exclusive_sum(keep ? 1 : 0)
//       leader             simd_is_first()
//       broadcast          simd_broadcast_first(j)
//     They are defined over the active threads of the SIMD-group. Apple does not say whether
//     the active threads inside a branch are exactly those that took it (C3), so every
//     thread calls them, in range or not, outside any branch. Then simd_is_first() is lane 0
//     whether or not lane 0 keeps; the count > 0 test skips the atomic for a SIMD-group that
//     keeps nothing.
//   - The SIMD-group width is not written anywhere: the functions need no lane index.
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

// region: atomic
kernel void filter_atomic(device const uint*  input      [[buffer(0)]],
                          device uint*        output     [[buffer(1)]],
                          constant uint&      n          [[buffer(2)]],
                          constant uint&      threshold  [[buffer(3)]],
                          device atomic_uint* outputSize [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) {
        uint val = input[i];
        if (val < threshold) {
            // Returns the old count: that is this key's slot.
            uint j = atomic_fetch_add_explicit(outputSize, 1u, memory_order_relaxed);
            output[j] = val;
        }
    }
}
// endregion

// region: warp
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

    // Every thread of the SIMD-group calls these, in range or not.
    uint flag = keep ? 1u : 0u;
    uint rank = simd_prefix_exclusive_sum(flag);   // keeping threads below this one
    uint count = simd_sum(flag);                   // keeping threads in the SIMD-group

    uint j = 0;
    if (simd_is_first() && count > 0) {
        j = atomic_fetch_add_explicit(outputSize, count, memory_order_relaxed);
    }
    j = simd_broadcast_first(j);                   // every thread learns where the range starts

    if (keep) {
        output[j + rank] = val;
    }
}
// endregion

// region: private
kernel void filter_privatised(device const uint*  input      [[buffer(0)]],
                              device uint*        output     [[buffer(1)]],
                              constant uint&      n          [[buffer(2)]],
                              constant uint&      threshold  [[buffer(3)]],
                              device atomic_uint* outputSize [[buffer(4)]],
                              uint i   [[thread_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
    // Initialize the private counter
    threadgroup uint output_s[BLOCK_DIM];
    threadgroup atomic_uint outputSize_s;
    if (tid == 0) {
        atomic_store_explicit(&outputSize_s, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Filter into the private list
    if (i < n) {
        uint val = input[i];
        if (val < threshold) {
            uint j = atomic_fetch_add_explicit(&outputSize_s, 1u, memory_order_relaxed);
            output_s[j] = val;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Reserve the threadgroup's range in the public list: one atomic per threadgroup
    threadgroup uint j_s;
    if (tid == 0) {
        uint count = atomic_load_explicit(&outputSize_s, memory_order_relaxed);
        j_s = atomic_fetch_add_explicit(outputSize, count, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Commit: thread k copies entry k, so neighbouring threads write neighbouring slots
    uint count = atomic_load_explicit(&outputSize_s, memory_order_relaxed);
    if (tid < count) {
        output[j_s + tid] = output_s[tid];
    }
}
// endregion
