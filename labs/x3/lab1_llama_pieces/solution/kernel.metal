// Lab X3.1 reference solution (Metal Shading Language). The same kernels as kernel.cu:
// rms_norm_rows, rope_rows and swiglu (part A), and rms_norm_port (part B, fixed). launch.mm
// dispatches them.
//
// CUDA                               Metal
// __shared__ float s[N]              threadgroup float* scratch [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// warpSize                           [[threads_per_simdgroup]]
// warp reduce (Fig. 10.13)           simd_sum(v): every active lane gets the sum
// expf, sqrtf                        exp, sqrt
// reinterpret_cast<const float4*>    the same buffer bound a second time as device float4*
//
// Notes:
//   - simd_sum's order of additions is not documented, so the last bits need not match a CUDA
//     tree (ch10-facts A4, C1). The README's tolerance allows for it.
//   - Under fast math, which may be this runtime's default (ch05-facts C7), sqrt is computed as
//     x * rsqrt(x) and x / y is within 2.5 ulp (ch21-facts C3); the behaviour of infinity is
//     undefined (ch10-facts C3), which is why swiglu below never forms one.
//   - The lane and SIMD-group index come from attributes, not from t / width (ch04-facts C2).
#include <metal_stdlib>
using namespace metal;

// Lab 10.3's block sum: stage 1 in every SIMD-group, stage 2 in SIMD-group 0 with one lane
// per SIMD-group's result (launch.mm checks there are no more groups than lanes), the result
// handed to every thread through scratch[0] behind a second barrier.
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

// ================================================================ part A
// ---------------------------------------------------------------- RMSNorm
// One threadgroup per row. y may be the same buffer as x (see kernel.cu).
kernel void rms_norm_rows(device const float* x   [[buffer(0)]],
                          device const float* w   [[buffer(1)]],
                          device float*       y   [[buffer(2)]],
                          constant uint&      d   [[buffer(3)]],
                          constant float&     eps [[buffer(4)]],
                          threadgroup float*  scratch [[threadgroup(0)]],
                          uint t      [[thread_position_in_threadgroup]],
                          uint row    [[threadgroup_position_in_grid]],
                          uint bdim   [[threads_per_threadgroup]],
                          uint lane   [[thread_index_in_simdgroup]],
                          uint sg     [[simdgroup_index_in_threadgroup]],
                          uint groups [[simdgroups_per_threadgroup]]) {
    device const float* in = x + ulong(row) * d;
    device float* out = y + ulong(row) * d;

    float s = 0.0f;
    for (uint c = t; c < d; c += bdim) {
        const float v = in[c];
        s += v * v;
    }
    s = block_sum(s, scratch, lane, sg, groups);

    const float scale = 1.0f / sqrt(s / float(d) + eps);   // llama.cpp Metal's form (x3-facts B2)
    for (uint c = t; c < d; c += bdim) {
        out[c] = in[c] * scale * w[c];
    }
}

// ---------------------------------------------------------------- RoPE
// One thread per rotated pair, as kernel.cu.
kernel void rope_rows(device float*       x       [[buffer(0)]],
                      device const int*   pos     [[buffer(1)]],
                      device const float* cosT    [[buffer(2)]],
                      device const float* sinT    [[buffer(3)]],
                      constant uint&      rows    [[buffer(4)]],
                      constant uint&      heads   [[buffer(5)]],
                      constant uint&      headDim [[buffer(6)]],
                      constant uint&      mode    [[buffer(7)]],
                      uint p [[thread_position_in_grid]]) {
    const uint half_ = headDim / 2;
    if (ulong(p) >= ulong(rows) * heads * half_) {
        return;
    }
    const uint i = p % half_;
    const uint head = p / half_;
    const uint r = head / heads;
    device float* v = x + ulong(head) * headDim;
    const uint a = mode == 0 ? 2 * i : i;
    const uint b = mode == 0 ? 2 * i + 1 : i + half_;
    const ulong t = ulong(pos[r]) * half_ + i;
    const float c = cosT[t];
    const float s = sinT[t];
    const float x0 = v[a];
    const float x1 = v[b];
    v[a] = x0 * c - x1 * s;
    v[b] = x0 * s + x1 * c;
}

// ---------------------------------------------------------------- SwiGLU
// gate / (1 + exp(-gate)) * up, written so that no intermediate overflows: with e = exp(-|g|)
// in (0, 1], sigmoid(g) is 1 / (1 + e) for g >= 0 and e / (1 + e) for g < 0. The CUDA file
// writes llama.cpp's form, which relies on exp(100) = infinity and g / infinity = 0; under
// Metal's fast math that behaviour is undefined (ch10-facts C3).
kernel void swiglu(device const float* gate [[buffer(0)]],
                   device const float* up   [[buffer(1)]],
                   device float*       out  [[buffer(2)]],
                   constant uint&      n    [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    if (i >= n) {
        return;
    }
    const float g = gate[i];                 // read before the write: out may be gate
    const float e = exp(-abs(g));
    const float sig = g >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    out[i] = g * sig * up[i];
}

// ================================================================ part B: the port, fixed
// llama.cpp's Metal RMSNorm in a row loop (x3-facts B2): SIMD-group 0 zeroes the slots, a
// strided sum, simd_sum, lane 0 of each SIMD-group writes its slot, SIMD-group 0 sums the
// slots. The float4 path reads the same buffers through their float4 bindings (Metal's
// documented way to read four floats at once, infra-facts C7); launch.mm chooses it, as
// llama.cpp's host chooses its _4 kernels.
kernel void rms_norm_port(device const float*  x    [[buffer(0)]],
                          device const float*  w    [[buffer(1)]],
                          device float*        y    [[buffer(2)]],
                          device const float4* x4   [[buffer(3)]],
                          device const float4* w4   [[buffer(4)]],
                          device float4*       y4   [[buffer(5)]],
                          constant uint&       rows [[buffer(6)]],
                          constant uint&       d    [[buffer(7)]],
                          constant float&      eps  [[buffer(8)]],
                          constant uint&       vec  [[buffer(9)]],
                          threadgroup float*   s_sum [[threadgroup(0)]],
                          uint t    [[thread_position_in_threadgroup]],
                          uint tg   [[threadgroup_position_in_grid]],
                          uint ntg  [[threadgroups_per_grid]],
                          uint bdim [[threads_per_threadgroup]],
                          uint lane [[thread_index_in_simdgroup]],
                          uint sg   [[simdgroup_index_in_threadgroup]]) {
    const uint n4 = (d + 3) / 4;
    for (uint row = tg; row < rows; row += ntg) {
        // FIX (a): SIMD-group 0 zeroes every slot stage 2 reads, as llama.cpp's Metal kernel
        // does (x3-facts B2). launch.mm sizes s_sum to one slot per lane.
        if (sg == 0) {
            s_sum[lane] = 0.0f;
        }

        float sum = 0.0f;
        if (vec != 0) {
            device const float4* xr = x4 + ulong(row) * n4;
            for (uint i = t; i < n4; i += bdim) {
                sum += dot(xr[i], xr[i]);
            }
        } else {
            device const float* xr = x + ulong(row) * d;
            for (uint i = t; i < d; i += bdim) {
                sum += xr[i] * xr[i];
            }
        }
        sum = simd_sum(sum);
        threadgroup_barrier(mem_flags::mem_threadgroup);   // the zeros are in place first
        if (lane == 0) {
            s_sum[sg] = sum;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sg == 0) {
            sum = simd_sum(s_sum[lane]);
            if (lane == 0) {
                s_sum[0] = sum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float scale = 1.0f / sqrt(s_sum[0] / float(d) + eps);

        if (vec != 0) {
            device const float4* xr = x4 + ulong(row) * n4;
            device float4* yr = y4 + ulong(row) * n4;
            for (uint i = t; i < n4; i += bdim) {
                yr[i] = xr[i] * scale * w4[i];
            }
        } else {
            device const float* xr = x + ulong(row) * d;
            device float* yr = y + ulong(row) * d;
            for (uint i = t; i < d; i += bdim) {
                yr[i] = xr[i] * scale * w[i];
            }
        }
        // FIX (b): every thread has read s_sum[0] before the next row zeroes it.
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
