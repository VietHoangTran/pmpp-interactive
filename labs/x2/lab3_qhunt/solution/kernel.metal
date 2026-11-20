// Lab X2.3 reference solution (Metal Shading Language): the same four defects as kernel.cu,
// fixed (FIX 1 to 4 below; kernel.cu explains each). The integer path of kernel.cu:
// quantise the activations to Q8_1, then a Q4_0 x Q8_1 GEMV with an integer dot per block.
// Not in the book: module X2 is ours; the arithmetic is llama.cpp's CUDA path (x2-facts D3).
// llama.cpp's own Metal decode kernels do not take this path: they stay in float and fold the
// nibble shifts into the activations (D4). This file is the CUDA design written in MSL, so
// that the same four defects can be hunted on both.
//
// CUDA                               Metal
// __dp4a (sm_61 and later)           none: MSL has no integer dot product (x2-facts C1);
//                                    dot4_i8 below is four products
// __shfl_xor_sync(mask, v, o, 32)    simd_shuffle_xor on the 32-bit pattern (it is declared
//                                    for integer types, infra-facts C2)
// warpSum (shuffles and a mask)      simd_sum: the sum, given to every lane, no mask
// __shared__ float partial[..]       threadgroup float partial[..]
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// half bits -> float                 float(as_type<half>(bits)) (x2-facts C5)
//
// launch.mm makes every threadgroup a whole number of SIMD-groups of the pipeline's
// threadExecutionWidth and refuses widths that are not a multiple of 32 for the quantiser
// (its 32-value groups must not straddle two SIMD-groups).
#include <metal_stdlib>
using namespace metal;

#define MAX_BATCH 8
#define WARPS 4          // the most SIMD-groups per threadgroup launch.mm dispatches

struct block_q4_0 {      // llama.cpp's block_q4_0: 18 bytes, alignment 2 (C5)
    ushort d;            // FP16 bits of the scale
    uchar qs[16];        // byte j: weight j (low nibble), weight j + 16 (high nibble)
};
struct block_q8_1 {      // llama.cpp's block_q8_1: 36 bytes
    ushort d;            // FP16 bits of amax / 127
    ushort s;            // FP16 bits of d * (sum of the codes)
    char qs[32];
};

// ================================================================ provided, correct
static inline int dot4_i8(int a, int b, int c) {
    const char4 va = as_type<char4>(a), vb = as_type<char4>(b);
    return c + int(va.x) * int(vb.x) + int(va.y) * int(vb.y) + int(va.z) * int(vb.z) + int(va.w) * int(vb.w);
}

// One thread per value, n % 32 == 0. Threads past n take part in the shuffles and store nothing.
kernel void quantize_q8_1(device const float* X  [[buffer(0)]],
                          device block_q8_1*  Xq [[buffer(1)]],
                          constant uint&      n  [[buffer(2)]],
                          uint i [[thread_position_in_grid]]) {
    const float x = i < n ? X[i] : 0.0f;
    float amax = fabs(x);
    for (ushort offset = 16; offset > 0; offset /= 2) {
        amax = max(amax, as_type<float>(simd_shuffle_xor(as_type<uint>(amax), offset)));
    }
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : int(round(x / d));
    int sum = q;
    for (ushort offset = 16; offset > 0; offset /= 2) {
        sum += simd_shuffle_xor(sum, offset);
    }
    if (i < n) {
        device block_q8_1& blk = Xq[i / 32];
        blk.qs[i % 32] = char(q);
        if (i % 32 == 0) {
            blk.d = as_type<ushort>(half(d));
            blk.s = as_type<ushort>(half(float(sum) * d));
        }
    }
}

// ================================================================ the GEMV
// One threadgroup of up to WARPS SIMD-groups per output row; thread t takes blocks t,
// t + threads, ... of the row.
kernel void gemv_q4_0_q8_1(device const block_q4_0* W     [[buffer(0)]],
                           device const block_q8_1* Xq    [[buffer(1)]],
                           device float*            Y     [[buffer(2)]],
                           constant uint&           batch [[buffer(3)]],
                           constant uint&           in    [[buffer(4)]],
                           constant uint&           out   [[buffer(5)]],
                           uint row     [[threadgroup_position_in_grid]],
                           uint t       [[thread_position_in_threadgroup]],
                           uint threads [[threads_per_threadgroup]],
                           uint lane    [[thread_index_in_simdgroup]],
                           uint sg      [[simdgroup_index_in_threadgroup]],
                           uint groups  [[simdgroups_per_threadgroup]]) {
    threadgroup float partial[WARPS][MAX_BATCH];
    const uint nb = in / 32;
    device const block_q4_0* w = W + row * nb;

    float sum[MAX_BATCH];
    for (uint b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    for (uint kb = t; kb < nb; kb += threads) {
        const float dw = float(as_type<half>(w[kb].d));
        device const ushort* q16 = (device const ushort*)w[kb].qs;    // 2-byte aligned only
        uint v[4];
        for (uint i = 0; i < 4; ++i) {
            v[i] = uint(q16[2 * i]) | (uint(q16[2 * i + 1]) << 16);
        }
        for (uint b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                device const block_q8_1& xb = Xq[b * nb + kb];
                device const int* u = (device const int*)xb.qs;       // offset 4 of a 36-byte block
                int sumi = 0;                                         // FIX 2: 32 bits, not a short
                for (uint i = 0; i < 4; ++i) {
                    const int vi0 = int((v[i] >> 0) & 0x0F0F0F0Fu);   // weights 4i .. 4i + 3
                    const int vi1 = int((v[i] >> 4) & 0x0F0F0F0Fu);   // weights 4i + 16 .. 4i + 19
                    sumi = dot4_i8(vi0, u[i], sumi);                  // FIX 1: words i and i + 4,
                    sumi = dot4_i8(vi1, u[i + 4], sumi);              // not 2i and 2i + 1
                }
                // FIX 3: the -8 comes out through s = d_x * (sum of the activation codes), not d_x.
                sum[b] += dw * (float(sumi) * float(as_type<half>(xb.d)) - 8.0f * float(as_type<half>(xb.s)));
            }
        }
    }

    for (uint b = 0; b < MAX_BATCH; ++b) {
        if (b < batch) {
            const float s = simd_sum(sum[b]);
            if (lane == 0) {
                partial[sg][b] = s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);       // FIX 4: every SIMD-group's partials are written
    if (sg == 0) {
        for (uint b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                const float s = simd_sum(lane < groups ? partial[lane][b] : 0.0f);
                if (lane == 0) {
                    Y[b * out + row] = s;
                }
            }
        }
    }
}
