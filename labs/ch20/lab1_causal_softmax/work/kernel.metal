// Lab 20.1 starter (Metal Shading Language). The same kernel as kernel.cu: causal_softmax,
// one threadgroup of 256 threads per row, grid x = rows, grid y = items. launch.mm
// dispatches it and sizes the per-SIMD-group scratch.
//
// CUDA                               Metal
// __shared__ float s[MAX_WARPS]      threadgroup float* scratch [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// blockIdx.x, blockIdx.y             [[threadgroup_position_in_grid]] .x, .y
// warp reduce + block reduce         simd_max / simd_sum, then threadgroup memory
// expf, logf                         exp, log
//
// No infinity anywhere: under fast math, which may be this runtime's default, its
// behaviour is undefined (notes/today/ch20-facts.md C5). The max identity is -FLT_MAX, and
// the mask is the loop bound, so a masked score is never read.
#include <metal_stdlib>
using namespace metal;

// ======================================================================================
// From lab 10.3's reference solution (milestone M4), unchanged: the two-stage block
// reductions. Called by every thread; they return the threadgroup's result to every
// thread. Use them; do not edit them.
// ======================================================================================
inline float block_max(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
    v = simd_max(v);                                  // stage 1: every SIMD-group
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {                                    // stage 2: SIMD-group 0
        v = lane < groups ? scratch[lane] : -FLT_MAX;
        v = simd_max(v);
        if (lane == 0) {
            scratch[0] = v;                           // hand the result to the threadgroup
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}

inline float block_sum(float v, threadgroup float* scratch, uint lane, uint sg, uint groups) {
    v = simd_sum(v);
    if (lane == 0) {
        scratch[sg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        v = lane < groups ? scratch[lane] : 0.0f;
        v = simd_sum(v);
        if (lane == 0) {
            scratch[0] = v;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return scratch[0];
}
// ======================================================================================
// End of the M4 code from lab 10.3.
// ======================================================================================

struct CausalDims { uint nq; uint nk; float scale; };   // must match launch.mm

kernel void causal_softmax(device const float* S     [[buffer(0)]],
                           device float*       P     [[buffer(1)]],
                           device float*       L     [[buffer(2)]],
                           constant CausalDims& dims [[buffer(3)]],
                           threadgroup float*  scratch [[threadgroup(0)]],
                           uint2 tid    [[thread_position_in_threadgroup]],
                           uint2 tg     [[threadgroup_position_in_grid]],
                           uint2 tsize  [[threads_per_threadgroup]],
                           uint  lane   [[thread_index_in_simdgroup]],
                           uint  sg     [[simdgroup_index_in_threadgroup]],
                           uint  groups [[simdgroups_per_threadgroup]]) {
    // Metal wants the position attributes all scalar or all the same vector size, hence
    // uint2 for these three; the threadgroup is one-dimensional, so only .x is used.
    const uint t = tid.x, bdim = tsize.x;                    // CUDA's threadIdx.x, blockDim.x

    // TODO: the same seven steps as in kernel.cu, in Metal's vocabulary. The row is tg.x
    //       and the item tg.y; compute offsets in ulong.
}
