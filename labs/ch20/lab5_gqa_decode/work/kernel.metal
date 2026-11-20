// Lab 20.5 starter (Metal Shading Language): the grouped-query decode of kernel.cu.
// One threadgroup of 256 threads per (K/V head, sequence), serving its GQ query heads.
//
// GQ, the group size hQ / hKv, is a compile-time constant here as the template parameter
// is in kernel.cu: the driver compiles this file once per group size (1 to 8) with GQ
// defined as a preprocessor macro. Size your arrays with it.
//
// The kernel below is lab 20.4's reference per-head kernel with only its name changed: it
// computes ONE query head per threadgroup. Regroup it, following kernel.cu's TODOs 1 to 5.
// simd_sum is the warpSumAll of kernel.cu; every lane must reach every simd_sum call. The
// driver checks for 32-lane SIMD-groups. Keep the running max finite (M_INIT): fast math
// leaves infinities undefined.
#include <metal_stdlib>
using namespace metal;

#ifndef GQ
#error "GQ (query heads per K/V head) is defined by the driver when it builds the pipeline"
#endif

#define HEAD_DIM 128
#define SIMD_WIDTH 32
#define BLOCK_SIZE 256
#define N_WARPS (BLOCK_SIZE / SIMD_WIDTH)
#define D_PER_LANE (HEAD_DIM / SIMD_WIDTH)
#define M_INIT (-FLT_MAX / 2.0f)

kernel void decode_attention_grouped(device const float* q     [[buffer(0)]],
                                     device const float* Kc    [[buffer(1)]],
                                     device const float* Vc    [[buffer(2)]],
                                     device const int*   len   [[buffer(3)]],
                                     device float*       out   [[buffer(4)]],
                                     constant uint&      hQ    [[buffer(5)]],
                                     constant uint&      hKv   [[buffer(6)]],
                                     constant uint&      maxSeq [[buffer(7)]],
                                     constant float&     scale [[buffer(8)]],
                                     uint2 tg   [[threadgroup_position_in_grid]],
                                     uint  tid  [[thread_index_in_threadgroup]],
                                     uint  lane [[thread_index_in_simdgroup]],
                                     uint  w    [[simdgroup_index_in_threadgroup]]) {
    threadgroup float m_s[N_WARPS];                    // TODO 1
    threadgroup float D_s[N_WARPS];
    threadgroup float O_s[N_WARPS][HEAD_DIM];

    const uint h = tg.x;                               // TODO 2
    const uint b = tg.y;
    const uint kvh = h / (hQ / hKv);
    const uint n = (uint)len[b];

    device const float* qrow = q + ((ulong)b * hQ + h) * HEAD_DIM;   // TODO 3
    float qr[D_PER_LANE];
    for (uint i = 0; i < D_PER_LANE; ++i) {
        qr[i] = qrow[lane + i * SIMD_WIDTH];
    }
    float m = M_INIT;
    float D = 0.0f;
    float O[D_PER_LANE];
    for (uint i = 0; i < D_PER_LANE; ++i) {
        O[i] = 0.0f;
    }

    const uint rowStride = hKv * HEAD_DIM;
    device const float* Kb = Kc + (ulong)b * maxSeq * rowStride + (ulong)kvh * HEAD_DIM;
    device const float* Vb = Vc + (ulong)b * maxSeq * rowStride + (ulong)kvh * HEAD_DIM;

    for (uint t = w; t < n; t += N_WARPS) {           // TODO 4
        device const float* krow = Kb + (ulong)t * rowStride;
        float partial = 0.0f;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            partial += qr[i] * krow[lane + i * SIMD_WIDTH];
        }
        const float s = simd_sum(partial) * scale;

        const float mNew = max(m, s);
        const float alpha = exp(m - mNew);
        const float p = exp(s - mNew);
        D = D * alpha + p;
        device const float* vrow = Vb + (ulong)t * rowStride;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            O[i] = O[i] * alpha + p * vrow[lane + i * SIMD_WIDTH];
        }
        m = mNew;
    }

    if (lane == 0) {                                   // TODO 5
        m_s[w] = m;
        D_s[w] = D;
    }
    for (uint i = 0; i < D_PER_LANE; ++i) {
        O_s[w][lane + i * SIMD_WIDTH] = O[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float mAll = M_INIT;
    for (uint x = 0; x < N_WARPS; ++x) {
        mAll = max(mAll, m_s[x]);
    }
    float weight[N_WARPS];
    float DAll = 0.0f;
    for (uint x = 0; x < N_WARPS; ++x) {
        weight[x] = exp(m_s[x] - mAll);
        DAll += D_s[x] * weight[x];
    }
    device float* orow = out + ((ulong)b * hQ + h) * HEAD_DIM;
    for (uint dd = tid; dd < HEAD_DIM; dd += BLOCK_SIZE) {
        float o = 0.0f;
        for (uint x = 0; x < N_WARPS; ++x) {
            o += O_s[x][dd] * weight[x];
        }
        orow[dd] = o / DAll;
    }
}
