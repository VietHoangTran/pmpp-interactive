// Lab 10.2 reference solution (Metal): the four defects of the starter, fixed.
// The same kernel as kernel.cu (book Figs. 10.13, 10.16 and 10.18), in Metal terms:
// a SIMD-group is the counterpart of a warp, a threadgroup of a block.
//
// Where Metal differs from the CUDA file:
//   - simd_shuffle_down takes no mask and no width (infra-facts C2). Apple documents no
//     SIMD-group width, so the kernel reads it from [[threads_per_simdgroup]] (C5).
//   - The per-SIMD-group array cannot be sized for "the narrowest warp" as in kernel.cu,
//     because no narrowest width is documented. It has one slot per thread, enough for
//     any width; launch.mm checks that one SIMD-group has a lane for every SIMD-group.
//   - The commit needs an atomic type: `device atomic_float*` and
//     atomic_fetch_add_explicit, Metal 3 and Apple7 or later (infra-facts C1).
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 128        // threads per threadgroup (main.cpp passes the value it launches)
#endif

// Fig. 10.13 with Metal's shuffle. Every lane of the SIMD-group must call it; lane 0
// returns the SIMD-group's sum. The upper `stride` lanes keep their own value (C2), so
// they compute sums nobody reads.
static float simd_reduce(float val, uint simd_size) {
    float partialSum = val;
    for (uint stride = simd_size / 2; stride > 0; stride /= 2) {
        partialSum += simd_shuffle_down(partialSum, ushort(stride));
    }
    return partialSum;
}

kernel void reduce_any(device const float*  in  [[buffer(0)]],
                       device atomic_float* out [[buffer(1)]],   // FIX 4 (see below)
                       constant uint&       n   [[buffer(2)]],
                       uint t           [[thread_position_in_threadgroup]],
                       uint block       [[threadgroup_position_in_grid]],
                       uint lane        [[thread_index_in_simdgroup]],
                       uint simdgroup   [[simdgroup_index_in_threadgroup]],
                       uint simd_size   [[threads_per_simdgroup]],
                       uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float partials_s[BLOCK_DIM];   // one slot per SIMD-group, room for any width

    uint segment = 2 * BLOCK_DIM * block;
    uint i = segment + t;

    // FIX 1: no thread leaves early. Reading a lane that has ended is undefined
    // ("A thread needs to only read data from another active thread", C2). A thread
    // past the end contributes 0 and stays for the shuffles.
    float partialSum = 0.0f;
    if (i < n) partialSum += in[i];
    if (i + BLOCK_DIM < n) partialSum += in[i + BLOCK_DIM];

    // Stage 1: every SIMD-group reduces its own values in registers.
    partialSum = simd_reduce(partialSum, simd_size);
    if (lane == 0) {
        partials_s[simdgroup] = partialSum;
    }
    // FIX 3: SIMD-groups "make independent forward progress" without a threadgroup
    // barrier (C5); this one orders the writes above before the reads below.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Stage 2: SIMD-group 0 reduces the per-SIMD-group sums.
    if (simdgroup == 0) {
        // FIX 2: only the first simd_groups slots were written; the starter read every
        // lane's slot, and unwritten threadgroup memory holds whatever was there.
        float blockSum = (lane < simd_groups) ? partials_s[lane] : 0.0f;
        blockSum = simd_reduce(blockSum, simd_size);
        if (t == 0) {
            // FIX 4: every threadgroup adds into the same *out; a plain += on a
            // `device float*` can lose another threadgroup's sum.
            atomic_fetch_add_explicit(out, blockSum, memory_order_relaxed);
        }
    }
}
