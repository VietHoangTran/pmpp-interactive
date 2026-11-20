// Lab 20.5 reference solution (Metal Shading Language): kernel.cu's grouped-query decode
// in Metal's vocabulary. One threadgroup of 256 threads per (K/V head, sequence) serves
// its g_q = hQ / hKv <= MAX_GQ query heads; each SIMD-group loads every K and V row it
// visits once and keeps g_q running states. simd_sum gives each score to every lane
// (ch20-facts C3); threadgroup arrays are declared in the kernel (C2); the running max is
// finite because fast math leaves infinities undefined (C5). The driver checks that the
// pipeline's SIMD-groups have 32 lanes.
#include <metal_stdlib>
using namespace metal;

#define HEAD_DIM 128
#define SIMD_WIDTH 32                          // the book's WARP_SIZE; checked against the pipeline
#define BLOCK_SIZE 256
#define N_WARPS (BLOCK_SIZE / SIMD_WIDTH)
#define D_PER_LANE (HEAD_DIM / SIMD_WIDTH)
#define M_INIT (-FLT_MAX / 2.0f)
#define MAX_GQ 8

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
    threadgroup float m_s[N_WARPS][MAX_GQ];
    threadgroup float D_s[N_WARPS][MAX_GQ];
    threadgroup float O_s[N_WARPS][HEAD_DIM];

    const uint kvh = tg.x;
    const uint b = tg.y;
    const uint gq = hQ / hKv;
    const uint h0 = kvh * gq;
    const uint n = (uint)len[b];

    float qr[MAX_GQ][D_PER_LANE];
    float m[MAX_GQ], D[MAX_GQ], O[MAX_GQ][D_PER_LANE];
    for (uint g = 0; g < MAX_GQ; ++g) {
        for (uint i = 0; i < D_PER_LANE; ++i) {
            qr[g][i] = 0.0f;
            O[g][i] = 0.0f;
        }
        if (g < gq) {
            device const float* qrow = q + ((ulong)b * hQ + h0 + g) * HEAD_DIM;
            for (uint i = 0; i < D_PER_LANE; ++i) {
                qr[g][i] = qrow[lane + i * SIMD_WIDTH];
            }
        }
        m[g] = M_INIT;
        D[g] = 0.0f;
    }

    const uint rowStride = hKv * HEAD_DIM;
    device const float* Kb = Kc + (ulong)b * maxSeq * rowStride + (ulong)kvh * HEAD_DIM;
    device const float* Vb = Vc + (ulong)b * maxSeq * rowStride + (ulong)kvh * HEAD_DIM;

    for (uint t = w; t < n; t += N_WARPS) {
        float kr[D_PER_LANE], vr[D_PER_LANE];
        device const float* krow = Kb + (ulong)t * rowStride;
        device const float* vrow = Vb + (ulong)t * rowStride;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            kr[i] = krow[lane + i * SIMD_WIDTH];
            vr[i] = vrow[lane + i * SIMD_WIDTH];
        }
        for (uint g = 0; g < MAX_GQ; ++g) {
            if (g < gq) {                              // the same in every lane of the SIMD-group
                float partial = 0.0f;
                for (uint i = 0; i < D_PER_LANE; ++i) {
                    partial += qr[g][i] * kr[i];
                }
                const float s = simd_sum(partial) * scale;
                const float mNew = max(m[g], s);
                const float alpha = exp(m[g] - mNew);
                const float p = exp(s - mNew);
                D[g] = D[g] * alpha + p;
                for (uint i = 0; i < D_PER_LANE; ++i) {
                    O[g][i] = O[g][i] * alpha + p * vr[i];
                }
                m[g] = mNew;
            }
        }
    }

    if (lane == 0) {
        for (uint g = 0; g < MAX_GQ; ++g) {
            m_s[w][g] = m[g];
            D_s[w][g] = D[g];
        }
    }
    for (uint g = 0; g < MAX_GQ; ++g) {
        if (g < gq) {                                  // the same in every thread: the barriers pair up
            for (uint i = 0; i < D_PER_LANE; ++i) {
                O_s[w][lane + i * SIMD_WIDTH] = O[g][i];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);

            float mAll = M_INIT;
            for (uint x = 0; x < N_WARPS; ++x) {
                mAll = max(mAll, m_s[x][g]);
            }
            float weight[N_WARPS];
            float DAll = 0.0f;
            for (uint x = 0; x < N_WARPS; ++x) {
                weight[x] = exp(m_s[x][g] - mAll);
                DAll += D_s[x][g] * weight[x];
            }
            device float* orow = out + ((ulong)b * hQ + h0 + g) * HEAD_DIM;
            for (uint dd = tid; dd < HEAD_DIM; dd += BLOCK_SIZE) {
                float o = 0.0f;
                for (uint x = 0; x < N_WARPS; ++x) {
                    o += O_s[x][dd] * weight[x];
                }
                orow[dd] = o / DAll;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
    }
}
