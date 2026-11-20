// Lab 11.3 reference solution (Metal): the three defects of the starter, fixed.
// The same kernel as kernel.cu, in Metal terms: a SIMD-group is the counterpart of a warp,
// a threadgroup of a block.
//
// Where Metal differs from the CUDA file:
//   - simd_shuffle_up takes no mask and no width, and leaves the lower `delta` lanes
//     unchanged, as CUDA does (notes/today/ch11-facts.md C1). Apple documents no SIMD-group
//     width, so the kernel reads it from [[threads_per_simdgroup]] (infra-facts C5).
//   - A helper function cannot declare threadgroup memory (ch11-facts C2), and cannot read
//     the thread attributes either: the lane index is passed to it.
//   - Lane and SIMD-group indices come from their own attributes, not from the position in
//     the threadgroup (ch11-facts C2); every position below is built from them.
//   - The warp barrier is simdgroup_barrier, the block barrier threadgroup_barrier, both with
//     mem_flags::mem_threadgroup (infra-facts C4, ch05-facts C2).
//   - The SIMD-group sums array has one slot per thread, enough for any width, because no
//     narrowest width is documented; launch.mm checks that one SIMD-group has a lane for
//     every SIMD-group.
#include <metal_stdlib>
using namespace metal;

#define COARSE_FACTOR 4      // elements per thread per chunk (the book's example factor, p. 269)
#define MAX_BLOCK_DIM 1024   // the largest threadgroup launch.mm accepts

// Fig. 11.8 with Metal's shuffle: inclusive Kogge-Stone scan across one SIMD-group. Every
// lane must call it.
static float simd_scan_inclusive(float val, uint lane, uint simd_size) {
    for (uint stride = 1; stride < simd_size; stride *= 2) {
        float leftVal = simd_shuffle_up(val, ushort(stride));
        if (lane >= stride) {
            val += leftVal;
        }
    }
    return val;
}

kernel void exclusive_row_scan(device const float* in   [[buffer(0)]],
                               device float*       out  [[buffer(1)]],
                               constant uint&      cols [[buffer(2)]],
                               uint row         [[threadgroup_position_in_grid]],
                               uint lane        [[thread_index_in_simdgroup]],
                               uint simdgroup   [[simdgroup_index_in_threadgroup]],
                               uint simd_size   [[threads_per_simdgroup]],
                               uint simd_groups [[simdgroups_per_threadgroup]]) {
    threadgroup float stage_s[COARSE_FACTOR * MAX_BLOCK_DIM];   // each SIMD-group's slice
    threadgroup float simdSums_s[MAX_BLOCK_DIM];                // one slot per SIMD-group, reused

    device const float* rowIn = in + (ulong)row * cols;
    device float* rowOut = out + (ulong)row * cols;
    const uint chunkSize = COARSE_FACTOR * simd_size * simd_groups;
    const uint groupSlice = simdgroup * COARSE_FACTOR * simd_size;
    threadgroup float* myStage = stage_s + groupSlice;

    float carry = 0.0f;   // the sum of every earlier chunk of this row
    for (uint chunk = 0; chunk < cols; chunk += chunkSize) {
        // 1. Coalesced load: in step c the SIMD-group reads simd_size consecutive elements.
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            uint j = chunk + groupSlice + c * simd_size + lane;
            myStage[c * simd_size + lane] = (j < cols) ? rowIn[j] : 0.0f;
        }
        // FIX 3: each lane reads below what other lanes just wrote.
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 2. Exclusive sequential scan of this lane's run, in registers.
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            run[c] = threadSum;
            threadSum += myStage[lane * COARSE_FACTOR + c];
        }

        // 3. Scan the run totals across the SIMD-group.
        // FIX 2: the starter passed the thread's position in the threadgroup as the lane.
        float inclusive = simd_scan_inclusive(threadSum, lane, simd_size);
        float exclusive = simd_shuffle_up(inclusive, ushort(1));
        if (lane == 0) {
            exclusive = 0.0f;
        }
        if (lane == simd_size - 1) {
            simdSums_s[simdgroup] = inclusive;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 4. SIMD-group 0 scans the SIMD-group totals.
        if (simdgroup == 0) {
            float groupSum = (lane < simd_groups) ? simdSums_s[lane] : 0.0f;
            groupSum = simd_scan_inclusive(groupSum, lane, simd_size);
            if (lane < simd_groups) {
                simdSums_s[lane] = groupSum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // 5. Everything before my run.
        float before = carry + ((simdgroup > 0) ? simdSums_s[simdgroup - 1] : 0.0f) + exclusive;
        carry += simdSums_s[simd_groups - 1];
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            myStage[lane * COARSE_FACTOR + c] = before + run[c];
        }
        simdgroup_barrier(mem_flags::mem_threadgroup);

        // 6. Coalesced store.
        for (uint c = 0; c < COARSE_FACTOR; ++c) {
            uint j = chunk + groupSlice + c * simd_size + lane;
            if (j < cols) {
                rowOut[j] = myStage[c * simd_size + lane];
            }
        }
        // FIX 1: no SIMD-group may write the next chunk's total into simdSums_s while
        // another is still reading this chunk's. SIMD-groups "make independent forward
        // progress" between threadgroup barriers (infra-facts C5).
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
