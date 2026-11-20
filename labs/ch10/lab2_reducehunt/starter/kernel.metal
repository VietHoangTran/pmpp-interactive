// Lab 10.2 starter (Metal): the same kernel as kernel.cu, with the same FOUR defects.
// A SIMD-group is Metal's counterpart of a warp, a threadgroup of a block.
// Reading a lane that has ended, or threadgroup memory nobody has written yet, is
// undefined: your GPU may hand back something that happens to be right. The CPU
// emulator (which runs kernel.cu) will not.
//
// Where Metal differs from the CUDA file (not defects):
//   - simd_shuffle_down takes no mask and no width (infra-facts C2). Apple documents no
//     SIMD-group width, so the kernel reads it from [[threads_per_simdgroup]] (C5).
//   - The per-SIMD-group array cannot be sized for "the narrowest warp" as in kernel.cu,
//     because no narrowest width is documented. It has one slot per thread, enough for
//     any width; launch.mm checks that one SIMD-group has a lane for every SIMD-group.
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 128        // threads per threadgroup (main.cpp passes the value it launches)
#endif

// Fig. 10.13 with Metal's shuffle. Every lane of the SIMD-group must call it; lane 0
// returns the SIMD-group's sum.
static float simd_reduce(float val, uint simd_size) {
    float partialSum = val;
    for (uint stride = simd_size / 2; stride > 0; stride /= 2) {
        partialSum += simd_shuffle_down(partialSum, ushort(stride));
    }
    return partialSum;
}

kernel void reduce_any(device const float* in  [[buffer(0)]],
                       device float*       out [[buffer(1)]],
                       constant uint&      n   [[buffer(2)]],
                       uint t           [[thread_position_in_threadgroup]],
                       uint block       [[threadgroup_position_in_grid]],
                       uint lane        [[thread_index_in_simdgroup]],
                       uint simdgroup   [[simdgroup_index_in_threadgroup]],
                       uint simd_size   [[threads_per_simdgroup]],
                       uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float partials_s[BLOCK_DIM];   // one slot per SIMD-group, room for any width

    uint segment = 2 * BLOCK_DIM * block;
    uint i = segment + t;

    if (i >= n) return;                        // nothing to add here
    float partialSum = in[i];
    if (i + BLOCK_DIM < n) partialSum += in[i + BLOCK_DIM];

    // Stage 1: every SIMD-group reduces its own values in registers.
    partialSum = simd_reduce(partialSum, simd_size);
    if (lane == 0) {
        partials_s[simdgroup] = partialSum;
    }

    // Stage 2: SIMD-group 0 reduces the per-SIMD-group sums.
    if (simdgroup == 0) {
        float blockSum = partials_s[lane];
        blockSum = simd_reduce(blockSum, simd_size);
        if (t == 0) {
            *out += blockSum;                  // commit the threadgroup's total
        }
    }
}
