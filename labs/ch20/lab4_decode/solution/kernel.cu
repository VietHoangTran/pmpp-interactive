// Lab 20.4 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M7: append one token's K and V to the KV cache, then attend one new query per
// (sequence, head) over the cache.
//
// Book: section 20.4 (pp. 488-492, Fig. 20.6) says what a decode step computes: one new
// row of Q, K and V, K' and V' appended to the cache, then S' = Q'K^T, P' = softmax(S'),
// O' = P'V. The book prints no code for the cache, the append or a decode kernel (p. 492),
// so both kernels here are ours. Their parts come from section 20.5:
//   - the online softmax: a running max m, a running denominator D and an un-normalised
//     output O~ per row, updated one key at a time (Figs. 20.13 to 20.15, Eqs. 20.3 to 20.6),
//     divided once at the end (Fig. 20.16);
//   - the interleaved register layout of Fig. 20.10: lane L of a warp holds dimensions
//     L, L + 32, L + 64, L + 96 of the query, so each load of a key row is coalesced.
//
// Contract (README.md states it in full):
//   k_new, v_new  [batch, h_kv, HEAD_DIM]           q, out  [batch, h_q, HEAD_DIM]
//   Kc, Vc        [batch, max_seq, h_kv, HEAD_DIM]  (the Llama reference layout,
//                                                    notes/today/ch20-facts.md D3)
//   pos, len      [batch] int, 0 <= pos[b] < max_seq, 1 <= len[b] <= max_seq
//   h_q a multiple of h_kv; query head h reads K/V head h / (h_q / h_kv).
//   Row-major float32 throughout, batch first, raw device pointers.
//
// The design of the decode kernel (ours; MLX's Metal sdpa_vector kernel has the same
// shape, ch20-facts D4): one block of 256 threads per (sequence, query head). The block's
// 8 warps split the keys: warp w takes keys w, w + 8, w + 16, ... and keeps its own
// (m, D, O~). At the end the warps merge through shared memory with Eqs. 20.4 and 20.6.
// That merge is the "split-K" scheme FlashAttention-1 used across the warps of a block
// (footnote 7, p. 498; ch20-facts A7) and Flash-Decoding uses across blocks (A8).
// No mask: a single new query may see every cached token (Fig. 20.6 has none; Meta's
// Llama code builds a mask only when more than one token is processed, D3).
//
// Where this file differs from the page, and why:
//   - The book's running max starts at NEG_INFINITY, a name no CUDA header defines, and
//     IEEE gives NaN for (-inf) - (-inf) (ch20-facts A5). Here it starts at the finite
//     M_INIT, as llama.cpp does (-FLT_MAX/2) and MLX does (finite_min) (A5, C5): the merge
//     below meets warps that saw no key at all, and their m must not poison it.
//   - The book's WARP_SIZE is the hardware warp. Here it is a logical warp of 32 lanes:
//     every shuffle passes width WARP_SIZE, so on a 64-wide AMD wavefront two logical
//     warps share a wavefront and never read each other's lanes (ch20-facts B2).
//   - Fig. 20.12 gets each score from a WarpReduce plus a broadcast from lane 0. A
//     butterfly (__shfl_xor_sync) leaves the sum in every lane in one pass, which the
//     online update needs (the CUDA guide's own warp reduction, and llama.cpp's
//     warp_reduce, are butterflies: ch10-facts D1).
//   - The lane mask has its own type: HIP's _sync intrinsics static_assert a 64-bit mask
//     (infra-facts B3, ch20-facts B1).
#include <cfloat>

// The course's CUDA front-end check declares no CUDA Math API functions, so it is given
// these two device declarations. nvcc, hipcc and the emulators never see them.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
__device__ float fmaxf(float x, float y);
#endif

#define HEAD_DIM 128                           // the book's d (Fig. 20.9); a multiple of WARP_SIZE
#define WARP_SIZE 32                           // the book's WARP_SIZE: a logical warp of 32 lanes (see above)
#define BLOCK_SIZE 256                         // threads per (sequence, head)
#define N_WARPS (BLOCK_SIZE / WARP_SIZE)       // 8 warps, so warp w takes keys w, w + 8, ...
#define D_PER_LANE (HEAD_DIM / WARP_SIZE)      // 4 dimensions of q, k, v and O~ per lane
#define M_INIT (-FLT_MAX / 2.0f)               // finite start for a running max (ch20-facts A5)

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (Chapter 10, used in Figs. 20.10 to 20.16), for the logical warp.
__device__ int laneIdx() { return threadIdx.x % WARP_SIZE; }
__device__ int warpIdx() { return threadIdx.x / WARP_SIZE; }

// Butterfly sum: after log2(32) = 5 exchanges every lane holds the sum of all 32 lanes.
__device__ float warpSumAll(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(full, v, offset, WARP_SIZE);
    }
    return v;
}

// ---------------------------------------------------------------- KV append
// One thread per element of k_new (and of v_new): element e of sequence b's new row goes
// to row pos[b] of sequence b in the cache. A scatter by index, like llama.cpp's
// ggml_set_rows (ch20-facts D3). Nothing else in either cache is touched.
__global__ void kvAppendKernel(const float* k_new, const float* v_new, float* Kc, float* Vc,
                               const int* pos, int batch, int max_seq, int h_kv) {
    const int rowLen = h_kv * HEAD_DIM;                 // one token's keys across the K/V heads
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= batch * rowLen) {
        return;
    }
    const int b = i / rowLen;
    const int e = i % rowLen;
    const int p = pos[b];
    if (p < 0 || p >= max_seq) {
        return;                                         // outside the cache: write nothing
    }
    const size_t dst = ((size_t)b * max_seq + p) * rowLen + e;
    Kc[dst] = k_new[i];
    Vc[dst] = v_new[i];
}

void launchKvAppend(const float* k_new, const float* v_new, float* Kc, float* Vc,
                    const int* pos, int batch, int max_seq, int h_kv) {
    const int n = batch * h_kv * HEAD_DIM;
    if (n <= 0) {
        return;
    }
    kvAppendKernel<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(k_new, v_new, Kc, Vc, pos, batch, max_seq, h_kv);
}

// ---------------------------------------------------------------- decode attention
// Block (h, b): query head h of sequence b.
__global__ void decodeAttentionKernel(const float* q, const float* Kc, const float* Vc, const int* len,
                                      float* out, int h_q, int h_kv, int max_seq, float scale) {
    __shared__ float m_s[N_WARPS];                      // each warp's running max ...
    __shared__ float D_s[N_WARPS];                      // ... denominator ...
    __shared__ float O_s[N_WARPS][HEAD_DIM];            // ... and un-normalised output

    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int kvh = h / (h_q / h_kv);                   // the K/V head this query head reads
    const int lane = laneIdx();
    const int w = warpIdx();
    const int n = len[b];

    // This lane's four elements of q, in registers (Fig. 20.10's interleaving).
    const float* qrow = q + ((size_t)b * h_q + h) * HEAD_DIM;
    float qr[D_PER_LANE];
    for (int i = 0; i < D_PER_LANE; ++i) {
        qr[i] = qrow[lane + i * WARP_SIZE];
    }

    // This warp's running state over the keys it has seen so far: the set A of Eq. 20.4.
    // Empty set: D = 0 and O~ = 0 (Exercise 3), m = the finite stand-in for -infinity.
    float m = M_INIT;
    float D = 0.0f;
    float O[D_PER_LANE];
    for (int i = 0; i < D_PER_LANE; ++i) {
        O[i] = 0.0f;
    }

    const size_t rowStride = (size_t)h_kv * HEAD_DIM;   // from one cached token to the next
    const float* Kb = Kc + (size_t)b * max_seq * rowStride + (size_t)kvh * HEAD_DIM;
    const float* Vb = Vc + (size_t)b * max_seq * rowStride + (size_t)kvh * HEAD_DIM;

    for (int t = w; t < n; t += N_WARPS) {
        // Score of key t: each lane multiplies its four elements, the butterfly adds the
        // 32 partial sums. Every lane ends with the same s.
        const float* krow = Kb + t * rowStride;
        float partial = 0.0f;
        for (int i = 0; i < D_PER_LANE; ++i) {
            partial += qr[i] * krow[lane + i * WARP_SIZE];
        }
        const float s = warpSumAll(partial) * scale;

        // Add the one-key set B = {t} to A (Figs. 20.14, 20.15, 20.13 with B_c = 1):
        const float mNew = fmaxf(m, s);                 // m_{A u B}
        const float alpha = expf(m - mNew);             // e^(m_A - m_{A u B}): rescales the old state.
                                                        // (p. 502 calls D times this the D_B term of
                                                        // Eq. 20.4; it is the A term: notes, "Unclear".)
        const float p = expf(s - mNew);                 // e^(l - m_{A u B}): already against the new max
        D = D * alpha + p;
        const float* vrow = Vb + t * rowStride;
        for (int i = 0; i < D_PER_LANE; ++i) {
            O[i] = O[i] * alpha + p * vrow[lane + i * WARP_SIZE];
        }
        m = mNew;
    }

    // Merge the 8 warps' disjoint key sets: Eq. 20.4 for D and Eq. 20.6 for O~, both in the
    // un-normalised form (each multiplied through by D_{A u B}), then one division (Fig. 20.16).
    if (lane == 0) {
        m_s[w] = m;
        D_s[w] = D;
    }
    for (int i = 0; i < D_PER_LANE; ++i) {
        O_s[w][lane + i * WARP_SIZE] = O[i];
    }
    __syncthreads();                                    // every warp's state is in shared memory

    float mAll = M_INIT;
    for (int x = 0; x < N_WARPS; ++x) {
        mAll = fmaxf(mAll, m_s[x]);
    }
    float weight[N_WARPS];                              // e^(m_x - m_all): 0 for a warp with no keys
    float DAll = 0.0f;
    for (int x = 0; x < N_WARPS; ++x) {
        weight[x] = expf(m_s[x] - mAll);
        DAll += D_s[x] * weight[x];
    }
    float* orow = out + ((size_t)b * h_q + h) * HEAD_DIM;
    for (int dd = threadIdx.x; dd < HEAD_DIM; dd += blockDim.x) {
        float o = 0.0f;
        for (int x = 0; x < N_WARPS; ++x) {
            o += O_s[x][dd] * weight[x];
        }
        orow[dd] = o / DAll;                            // len >= 1, so DAll >= 1
    }
}

void launchDecodeAttention(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                           int batch, int h_q, int h_kv, int max_seq, float scale) {
    if (batch <= 0 || h_q <= 0) {
        return;
    }
    dim3 dimGrid(h_q, batch);                           // one block per (head, sequence)
    decodeAttentionKernel<<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale);
}
