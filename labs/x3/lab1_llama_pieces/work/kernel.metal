// Lab X3.1 starter (Metal Shading Language). The same kernels as kernel.cu: rms_norm_rows,
// rope_rows and swiglu (part A, yours to write), and rms_norm_port (part B, the port to
// review). launch.mm dispatches them.
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
//     undefined (ch10-facts C3). Keep that in mind for swiglu.
//   - The lane and SIMD-group index come from attributes, not from t / width (ch04-facts C2).
#include <metal_stdlib>
using namespace metal;

// ================================================================ part A
// TODO 1: lab 10.3's block_sum (bring yours): simd_sum in every SIMD-group, one slot per
//         SIMD-group in threadgroup memory, SIMD-group 0 sums the slots, every thread gets the
//         result. launch.mm checks there are no more SIMD-groups than lanes.

// ---------------------------------------------------------------- RMSNorm
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
    // TODO 2: kernel.cu's TODO 2. y may be the same buffer as x.
}

// ---------------------------------------------------------------- RoPE
kernel void rope_rows(device float*       x       [[buffer(0)]],
                      device const int*   pos     [[buffer(1)]],
                      device const float* cosT    [[buffer(2)]],
                      device const float* sinT    [[buffer(3)]],
                      constant uint&      rows    [[buffer(4)]],
                      constant uint&      heads   [[buffer(5)]],
                      constant uint&      headDim [[buffer(6)]],
                      constant uint&      mode    [[buffer(7)]],
                      uint p [[thread_position_in_grid]]) {
    // TODO 3: kernel.cu's TODO 4. launch.mm may dispatch more threads than pairs. (`half` is a
    //         type in Metal: do not name a variable that.)
}

// ---------------------------------------------------------------- SwiGLU
kernel void swiglu(device const float* gate [[buffer(0)]],
                   device const float* up   [[buffer(1)]],
                   device float*       out  [[buffer(2)]],
                   constant uint&      n    [[buffer(3)]],
                   uint i [[thread_position_in_grid]]) {
    // TODO 4: kernel.cu's TODO 6. The test feeds gate = -100. Under fast math the behaviour of
    //         infinity is undefined, so find a way to write gate * sigmoid(gate) in which no
    //         intermediate value overflows.
}

// ================================================================ part B: the port
// llama.cpp's Metal RMSNorm in a row loop (x3-facts B2), as a colleague ported it: a strided
// sum, simd_sum, lane 0 of each SIMD-group writes its slot, SIMD-group 0 sums the slots. The
// float4 path reads the same buffers through their float4 bindings (Metal's documented way to
// read four floats at once, infra-facts C7); launch.mm chooses it, as llama.cpp's host chooses
// its _4 kernels.
//
// It compiles. The port has THREE independent defects, the same three as kernel.cu's; on
// Metal one of them lives in launch.mm. Find and fix all three (README.md, part B).
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
    }
}
