// Lab 20.4 reference solution (Metal Shading Language). Milestone M7: the KV-cache append
// and the single-token decode attention of kernel.cu, in Metal's vocabulary. Layouts and
// the contract are the same (README.md).
//
// CUDA                               Metal
// __shared__ (in the kernel)         threadgroup (in the kernel: MSL allows threadgroup
//                                    variables only in kernel functions, ch20-facts C2)
// warp, threadIdx.x % 32             SIMD-group, [[thread_index_in_simdgroup]]
// threadIdx.x / 32                   [[simdgroup_index_in_threadgroup]]
// butterfly of __shfl_xor_sync       simd_sum: the sum, returned to every lane (C3)
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
//
// Apple does not document the SIMD-group width as a number (ch20-facts C3). This layout
// gives each lane HEAD_DIM / 32 = 4 dimensions, so it needs SIMD-groups of 32: the driver
// checks the pipeline's threadExecutionWidth and fails with a message otherwise. Under
// Metal's default fast math, infinities are undefined (C5), so the running max starts at
// a finite value, as llama.cpp (-FLT_MAX/2) and MLX (finite_min) do on Metal.
#include <metal_stdlib>
using namespace metal;

#define HEAD_DIM 128
#define SIMD_WIDTH 32                          // the book's WARP_SIZE; checked against the pipeline
#define BLOCK_SIZE 256                         // threads per threadgroup: 8 SIMD-groups
#define N_WARPS (BLOCK_SIZE / SIMD_WIDTH)
#define D_PER_LANE (HEAD_DIM / SIMD_WIDTH)
#define M_INIT (-FLT_MAX / 2.0f)

// ---------------------------------------------------------------- KV append
// One thread per element of k_new (and v_new); a scatter of each sequence's new row to
// row pos[b] of its cache (ggml_set_rows, ch20-facts D3).
kernel void kv_append(device const float* k_new [[buffer(0)]],
                      device const float* v_new [[buffer(1)]],
                      device float*       Kc    [[buffer(2)]],
                      device float*       Vc    [[buffer(3)]],
                      device const int*   pos   [[buffer(4)]],
                      constant uint&      batch [[buffer(5)]],
                      constant uint&      maxSeq [[buffer(6)]],
                      constant uint&      hKv   [[buffer(7)]],
                      uint i [[thread_position_in_grid]]) {
    const uint rowLen = hKv * HEAD_DIM;
    if (i >= batch * rowLen) {
        return;
    }
    const uint b = i / rowLen;
    const uint e = i % rowLen;
    const int p = pos[b];
    if (p < 0 || (uint)p >= maxSeq) {
        return;
    }
    const ulong dst = ((ulong)b * maxSeq + (uint)p) * rowLen + e;
    Kc[dst] = k_new[i];
    Vc[dst] = v_new[i];
}

// ---------------------------------------------------------------- decode attention
// Threadgroup (h, b): query head h of sequence b. SIMD-group w takes keys w, w + 8, ...
kernel void decode_attention(device const float* q     [[buffer(0)]],
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
    threadgroup float m_s[N_WARPS];
    threadgroup float D_s[N_WARPS];
    threadgroup float O_s[N_WARPS][HEAD_DIM];

    const uint h = tg.x;
    const uint b = tg.y;
    const uint kvh = h / (hQ / hKv);
    const uint n = (uint)len[b];

    device const float* qrow = q + ((ulong)b * hQ + h) * HEAD_DIM;
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

    for (uint t = w; t < n; t += N_WARPS) {
        device const float* krow = Kb + (ulong)t * rowStride;
        float partial = 0.0f;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            partial += qr[i] * krow[lane + i * SIMD_WIDTH];
        }
        const float s = simd_sum(partial) * scale;     // the same in every lane

        const float mNew = max(m, s);
        const float alpha = exp(m - mNew);             // rescales the old state (Eq. 20.4's A term)
        const float p = exp(s - mNew);
        D = D * alpha + p;
        device const float* vrow = Vb + (ulong)t * rowStride;
        for (uint i = 0; i < D_PER_LANE; ++i) {
            O[i] = O[i] * alpha + p * vrow[lane + i * SIMD_WIDTH];
        }
        m = mNew;
    }

    // Merge the SIMD-groups' disjoint key sets (Eqs. 20.4 and 20.6, un-normalised).
    if (lane == 0) {
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
