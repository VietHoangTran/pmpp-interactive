// Lab 10.3 reference solution (Metal Shading Language). The same three kernels as
// kernel.cu: row_max, softmax_rows, argmax_rows. One threadgroup of BLOCK_SIZE threads
// per row; launch.mm dispatches them.
//
// CUDA                               Metal
// __shared__ float s[MAX_WARPS]      threadgroup float* scratch [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// threadIdx.x % warpSize             [[thread_index_in_simdgroup]]
// threadIdx.x / warpSize             [[simdgroup_index_in_threadgroup]]
// warp_reduce with + (Fig. 10.13)    simd_sum(v)   every active lane gets the sum
// warp_reduce with max               simd_max(v)   every active lane gets the max
// __shfl_down_sync(mask, v, d)       simd_shuffle_down(v, d)   (no mask)
// expf                               exp
//
// Notes:
//   - Metal has float SIMD-group reductions in one call (simd_sum, simd_max; Apple7 and
//     later), where CUDA has none (notes/today/ch10-facts.md A4, C1). The order in which
//     simd_sum adds is not documented, so its last bits need not match a CUDA tree.
//   - The max identity is -FLT_MAX, not -INFINITY: under fast math, which may be this
//     runtime's default, the behaviour of infinity is undefined (ch10-facts C3, ch05-facts C7).
//   - exp's error under fast math is at most 3 + floor(|2x|) ulp (ch10-facts C4). Softmax
//     keeps x at or below 0; README.md sets the test tolerance from that bound.
//   - The split of a threadgroup into SIMD-groups is the implementation's, so the lane and
//     the SIMD-group index come from attributes, not from t / width (ch04-facts C2).
#include <metal_stdlib>
using namespace metal;

// Stage 2 runs in SIMD-group 0 with one lane per SIMD-group's result; launch.mm checks
// that there are no more SIMD-groups than lanes.

// ---------------------------------------------------------------- block level (Fig. 10.16)
// Called by every thread. Returns the threadgroup's result to every thread.
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

// Larger value wins; equal values go to the lower index. Commutative and associative on
// finite values.
inline void argmax_combine(thread float& v, thread int& i, float otherV, int otherI) {
    if (otherV > v || (otherV == v && otherI < i)) {
        v = otherV;
        i = otherI;
    }
}

// Value and index shuffled separately, as MLX's arg_reduce does (ch10-facts D3). Lane 0 ends
// with the SIMD-group's pair. (A Metal-only alternative, ours: m = simd_max(v), then
// simd_min of the indices of the lanes whose v == m.)
inline void simd_argmax(thread float& v, thread int& i, uint width) {
    for (uint stride = width / 2; stride > 0; stride /= 2) {
        float otherV = simd_shuffle_down(v, ushort(stride));
        int otherI = simd_shuffle_down(i, ushort(stride));
        argmax_combine(v, i, otherV, otherI);
    }
}

// ---------------------------------------------------------------- row max
kernel void row_max(device const float* x   [[buffer(0)]],
                    device float*       out [[buffer(1)]],
                    constant uint&      cols [[buffer(2)]],
                    threadgroup float*  scratch [[threadgroup(0)]],
                    uint t      [[thread_position_in_threadgroup]],
                    uint row    [[threadgroup_position_in_grid]],
                    uint bdim   [[threads_per_threadgroup]],
                    uint lane   [[thread_index_in_simdgroup]],
                    uint sg     [[simdgroup_index_in_threadgroup]],
                    uint groups [[simdgroups_per_threadgroup]]) {
    device const float* in = x + ulong(row) * cols;
    float m = -FLT_MAX;
    for (uint c = t; c < cols; c += bdim) {           // coalesced: one contiguous run per pass
        m = max(m, in[c]);
    }
    m = block_max(m, scratch, lane, sg, groups);
    if (t == 0) {
        out[row] = m;
    }
}

// ---------------------------------------------------------------- softmax
kernel void softmax_rows(device const float* x     [[buffer(0)]],
                         device float*       y     [[buffer(1)]],
                         constant uint&      cols  [[buffer(2)]],
                         constant float&     scale [[buffer(3)]],
                         threadgroup float*  scratch [[threadgroup(0)]],
                         uint t      [[thread_position_in_threadgroup]],
                         uint row    [[threadgroup_position_in_grid]],
                         uint bdim   [[threads_per_threadgroup]],
                         uint lane   [[thread_index_in_simdgroup]],
                         uint sg     [[simdgroup_index_in_threadgroup]],
                         uint groups [[simdgroups_per_threadgroup]]) {
    device const float* in = x + ulong(row) * cols;
    device float* out = y + ulong(row) * cols;

    float m = -FLT_MAX;
    for (uint c = t; c < cols; c += bdim) {
        m = max(m, scale * in[c]);
    }
    m = block_max(m, scratch, lane, sg, groups);

    // Every thread has just read m from scratch[0]; block_sum writes scratch[0] again.
    // Without this barrier SIMD-group 0 can overwrite m before another group has read it.
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float s = 0.0f;
    for (uint c = t; c < cols; c += bdim) {
        s += exp(scale * in[c] - m);
    }
    s = block_sum(s, scratch, lane, sg, groups);

    const float inv = 1.0f / s;
    for (uint c = t; c < cols; c += bdim) {
        out[c] = exp(scale * in[c] - m) * inv;
    }
}

// ---------------------------------------------------------------- argmax
kernel void argmax_rows(device const float* x    [[buffer(0)]],
                        device int*         idx  [[buffer(1)]],
                        constant uint&      cols [[buffer(2)]],
                        threadgroup float*  vals [[threadgroup(0)]],
                        threadgroup int*    idxs [[threadgroup(1)]],
                        uint t      [[thread_position_in_threadgroup]],
                        uint row    [[threadgroup_position_in_grid]],
                        uint bdim   [[threads_per_threadgroup]],
                        uint width  [[threads_per_simdgroup]],
                        uint lane   [[thread_index_in_simdgroup]],
                        uint sg     [[simdgroup_index_in_threadgroup]],
                        uint groups [[simdgroups_per_threadgroup]]) {
    device const float* in = x + ulong(row) * cols;
    float v = -FLT_MAX;
    int i = INT_MAX;                                  // loses every tie, even against -FLT_MAX
    for (uint c = t; c < cols; c += bdim) {
        argmax_combine(v, i, in[c], int(c));
    }

    simd_argmax(v, i, width);
    if (lane == 0) {
        vals[sg] = v;
        idxs[sg] = i;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sg == 0) {
        v = -FLT_MAX;
        i = INT_MAX;
        if (lane < groups) {
            v = vals[lane];
            i = idxs[lane];
        }
        simd_argmax(v, i, width);
        if (lane == 0) {
            idx[row] = i;
        }
    }
}
