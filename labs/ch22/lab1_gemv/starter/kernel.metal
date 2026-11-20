// Lab 22.1 starter (Metal Shading Language). The same two decompositions as kernel.cu,
// plus the zeroing kernel that encodeGemvSplitK dispatches first.
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major, batch <= MAX_BATCH.
//
// CUDA                               Metal
// warp                               SIMD-group
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// blockDim.x / warpSize              [[simdgroups_per_threadgroup]]
// warpSum (shuffles and a mask)      simd_sum: the sum, given to every lane, no mask
// atomicAdd(float*, float)           device atomic_float* and atomic_fetch_add_explicit
//
// Metal leaves the split of a threadgroup into SIMD-groups to the implementation, so the
// SIMD-group index and lane come from attributes (notes/today/ch04-facts.md C2, infra-facts
// C5). launch.mm makes every threadgroup a whole number of SIMD-groups of the pipeline's
// threadExecutionWidth, the SIMD-group width (ch03-facts C9), so no SIMD-group is partial.
// simd_sum needs an Apple7 GPU or later; float atomics need Metal 3 and Apple7 or later
// (infra-facts C2, C1).
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8

// ---------------------------------------------------------------- one SIMD-group, one row of W: yours
// TODO 4: the same as warpRowDots in kernel.cu, used by both kernels. One SIMD-group's dot
// products of a stretch [k0, k1) of row w of W with every batch row of X. Every lane calls
// it with the same arguments (and its own lane); on return lane 0's total[b] must hold the
// sum over k in [k0, k1) of X[b * in + k] * w[k], for every b < batch.
//   a. One running sum per batch row, MAX_BATCH floats, `b < batch` tested inside.
//   b. The lanes stride along [k0, k1) by `width`, reading w contiguously.
//   c. simd_sum adds a value across the SIMD-group and gives the sum to every lane.
void simdRowDots(device const float* X, device const float* w, uint batch, uint in, uint k0, uint k1,
                 uint lane, uint width, thread float* total) {
    for (uint b = 0; b < MAX_BATCH; ++b) {
        total[b] = 0.0f;   // a placeholder so the starter runs: replace it
    }
}

// ---------------------------------------------------------------- (a) gather: yours
kernel void gemv_gather(device const float* X     [[buffer(0)]],
                        device const float* W     [[buffer(1)]],
                        device float*       Y     [[buffer(2)]],
                        constant uint&      batch [[buffer(3)]],
                        constant uint&      in    [[buffer(4)]],
                        constant uint&      out   [[buffer(5)]],
                        uint tg     [[threadgroup_position_in_grid]],
                        uint width  [[threads_per_simdgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint sg     [[simdgroup_index_in_threadgroup]],
                        uint groups [[simdgroups_per_threadgroup]]) {
    // TODO 5: the same kernel as gemvGatherKernel in kernel.cu, one SIMD-group per output.
    //   a. This SIMD-group's output j comes from tg, groups and sg. Past the end: return.
    //   b. Call simdRowDots on the whole of row j.
    //   c. Lane 0 writes Y[b * out + j].
}

// ---------------------------------------------------------------- (b) split-K: provided
// This kernel has the same ONE defect as gemvSplitKKernel in kernel.cu. Metal has no
// race detector here; README, "What each backend can catch", says how the driver looks
// for it. The fix changes the type of one parameter; launch.mm binds the same buffer
// whatever its type in the kernel. It calls your simdRowDots on its chunk, so it computes
// nothing until TODO 4 is done.
//
// SIMD-group g takes output j = g / splits and chunk s = g % splits of `in`.
kernel void gemv_splitk(device const float* X      [[buffer(0)]],
                        device const float* W      [[buffer(1)]],
                        device float*       Y      [[buffer(2)]],
                        constant uint&      batch  [[buffer(3)]],
                        constant uint&      in     [[buffer(4)]],
                        constant uint&      out    [[buffer(5)]],
                        constant uint&      splits [[buffer(6)]],
                        uint tg     [[threadgroup_position_in_grid]],
                        uint width  [[threads_per_simdgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint sg     [[simdgroup_index_in_threadgroup]],
                        uint groups [[simdgroups_per_threadgroup]]) {
    const uint g = tg * groups + sg;
    if (g >= splits * out) {
        return;
    }
    const uint j = g / splits;
    const uint s = g % splits;
    const uint chunk = (in + splits - 1) / splits;
    const uint k0 = s * chunk;
    const uint k1 = min(k0 + chunk, in);
    if (k0 >= in) {
        return;                         // an empty chunk: nothing to add
    }
    float total[MAX_BATCH];
    simdRowDots(X, W + j * in, batch, in, k0, k1, lane, width, total);   // chunk s of row j only
    if (lane == 0) {
        for (uint b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                Y[b * out + j] += total[b];
            }
        }
    }
}

// ---------------------------------------------------------------- provided: Y = 0
// The counterpart of launchGemvSplitK's cudaMemset. encodeGemvSplitK dispatches it in the
// same serial encoder, just before gemv_splitk.
kernel void gemv_zero(device float*  Y [[buffer(0)]],
                      constant uint& n [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) {
        Y[i] = 0.0f;
    }
}
