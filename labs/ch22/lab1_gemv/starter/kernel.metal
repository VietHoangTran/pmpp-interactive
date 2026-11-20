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
    // TODO 3: the same kernel as gemvGatherKernel in kernel.cu, one SIMD-group per output.
    //   a. This SIMD-group's output j comes from tg, groups and sg. Past the end: return.
    //   b. One running sum per batch row, MAX_BATCH floats, `b < batch` tested inside.
    //   c. The lanes stride along `in` by `width`, reading row j of W contiguously.
    //   d. simd_sum adds a value across the SIMD-group and gives the sum to every lane.
    //   e. Lane 0 writes Y[b * out + j].
}

// ---------------------------------------------------------------- (b) split-K: provided
// This kernel has the same ONE defect as gemvSplitKKernel in kernel.cu. Metal has no
// race detector here; README, "What each backend can catch", says how the driver looks
// for it. The fix changes the type of one parameter; launch.mm binds the same buffer
// whatever its type in the kernel.
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
    device const float* w = W + j * in;

    float sum[MAX_BATCH];
    for (uint b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    for (uint k = k0 + lane; k < k1; k += width) {
        const float wk = w[k];
        for (uint b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                sum[b] += X[b * in + k] * wk;
            }
        }
    }
    for (uint b = 0; b < MAX_BATCH; ++b) {
        if (b < batch) {
            const float total = simd_sum(sum[b]);
            if (lane == 0) {
                Y[b * out + j] += total;
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
