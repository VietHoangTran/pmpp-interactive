// Lab 20.1 reference solution (Metal Shading Language). The same kernel as kernel.cu:
// causal_softmax, one threadgroup of 256 threads per row, grid x = rows, grid y = items.
// launch.mm dispatches it and sizes the per-SIMD-group scratch.
//
// CUDA                               Metal
// __shared__ float s[MAX_WARPS]      threadgroup float* scratch [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// blockIdx.x, blockIdx.y             [[threadgroup_position_in_grid]] .x, .y
// warp reduce + block reduce         simd_max / simd_sum, then threadgroup memory
// expf, logf                         exp, log
//
// Notes:
//   - No infinity anywhere: the max identity is -FLT_MAX and masked scores are never read
//     (the loop bound is the mask, as in the book). Under fast math, which may be this
//     runtime's default, the behaviour of infinity is undefined (notes/today/ch20-facts.md
//     C5, ch10-facts C3, ch05-facts C7).
//   - simd_max and simd_sum return the result to every active lane (infra-facts C2), so
//     Metal needs no shuffle tree for stage 1 or stage 2.
//   - The split of a threadgroup into SIMD-groups is the implementation's, so the lane and
//     the SIMD-group index come from attributes, not from t / width (ch04-facts C2).
#include <metal_stdlib>
using namespace metal;

// ======================================================================================
// From lab 10.3's reference solution (milestone M4), unchanged: the two-stage block
// reductions. Called by every thread; they return the threadgroup's result to every
// thread. Stage 2 gives each SIMD-group's result one lane of SIMD-group 0; launch.mm
// checks that there are no more SIMD-groups than lanes.
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
    const uint nq = dims.nq, nk = dims.nk;
    const float scale = dims.scale;
    // Metal wants the position attributes all scalar or all the same vector size, hence
    // uint2 for these three; the threadgroup is one-dimensional, so only .x is used.
    const uint t = tid.x, bdim = tsize.x;                    // CUDA's threadIdx.x, blockDim.x
    const uint row = tg.x;                                   // CUDA's blockIdx.x
    const ulong item = tg.y;                                 // CUDA's blockIdx.y
    const ulong rowStart = (item * nq + row) * ulong(nk);
    device const float* S_row = S + rowStart;
    device float* P_row = P + rowStart;

    // Columns 0 .. visible-1 are unmasked (lower-right alignment, nq <= nk).
    const uint visible = row + (nk - nq) + 1;

    float m = -FLT_MAX;
    for (uint c = t; c < visible; c += bdim) {
        m = max(m, scale * S_row[c]);
    }
    m = block_max(m, scratch, lane, sg, groups);

    // Every thread has just read m from scratch[0]; block_sum writes scratch[0] again.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = 0.0f;
    for (uint c = t; c < visible; c += bdim) {
        s += exp(scale * S_row[c] - m);
    }
    s = block_sum(s, scratch, lane, sg, groups);

    const float inv = 1.0f / s;
    for (uint c = t; c < nk; c += bdim) {
        if (c < visible) {
            P_row[c] = exp(scale * S_row[c] - m) * inv;
        } else {
            P_row[c] = 0.0f;
        }
    }

    if (t == 0) {
        L[item * nq + row] = m + log(s);
    }
}
