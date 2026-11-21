// Lab 20.5 starter (CUDA; also built for HIP and for the CPU emulators).
// Grouped-query decode: one block per (K/V head, sequence) serving all g_q = h_q / h_kv
// query heads that read that K/V head, so that each K and V row is loaded once for g_q
// queries. Book: section 20.7 (pp. 508-509, Eq. 20.13). README.md has the details.
//
// Contract: lab 20.4's launchDecodeAttention exactly, plus g_q <= MAX_GQ.
//   q, out [batch, h_q, HEAD_DIM]; Kc, Vc [batch, max_seq, h_kv, HEAD_DIM]; len [batch],
//   1 <= len[b] <= max_seq; query head h reads K/V head h / g_q.
//
// The kernel body below is lab 20.4's reference per-head kernel: it computes ONE query head
// per block. It is already a template on GQ, the group size, so that the arrays you add can
// be sized GQ and every loop over them has a trip count the compiler knows: then they can
// stay in registers. Regroup the body: the TODOs mark where.
#include <cfloat>
#include <cstdio>

#define HEAD_DIM 128
#define WARP_SIZE 32                           // the book's WARP_SIZE: a logical warp of 32 lanes
#define BLOCK_SIZE 256
#define N_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_PER_LANE (HEAD_DIM / WARP_SIZE)
#define M_INIT (-FLT_MAX / 2.0f)
#define MAX_GQ 8                               // the largest group your launcher must serve

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ int laneIdx() { return threadIdx.x % WARP_SIZE; }
__device__ int warpIdx() { return threadIdx.x / WARP_SIZE; }

__device__ float warpSumAll(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        v += __shfl_xor_sync(full, v, offset, WARP_SIZE);
    }
    return v;
}

template <int GQ>
__global__ void decodeAttentionGroupedKernel(const float* q, const float* Kc, const float* Vc, const int* len,
                                             float* out, int h_q, int h_kv, int max_seq, float scale) {
    // TODO 1: the warps now merge GQ heads. Give m_s and D_s a second dimension of GQ.
    //         O_s can stay [N_WARPS][HEAD_DIM] if the merge handles one query head at a
    //         time (eight heads at once would be 32 KiB, the whole of a Metal
    //         threadgroup's memory).
    __shared__ float m_s[N_WARPS];
    __shared__ float D_s[N_WARPS];
    __shared__ float O_s[N_WARPS][HEAD_DIM];

    // TODO 2: the block is now (K/V head, sequence): blockIdx.x is the K/V head, and its
    //         query heads are kvh * GQ ... kvh * GQ + GQ - 1.
    const int h = blockIdx.x;
    const int b = blockIdx.y;
    const int kvh = h / (h_q / h_kv);
    const int lane = laneIdx();
    const int w = warpIdx();
    const int n = len[b];

    // TODO 3: GQ queries and GQ running states (m, D, O~) per warp: arrays of GQ, filled
    //         in loops over g < GQ (`#pragma unroll` on them is harmless and documents
    //         the intent).
    const float* qrow = q + ((size_t)b * h_q + h) * HEAD_DIM;
    float qr[D_PER_LANE];
    for (int i = 0; i < D_PER_LANE; ++i) {
        qr[i] = qrow[lane + i * WARP_SIZE];
    }
    float m = M_INIT;
    float D = 0.0f;
    float O[D_PER_LANE];
    for (int i = 0; i < D_PER_LANE; ++i) {
        O[i] = 0.0f;
    }

    const size_t rowStride = (size_t)h_kv * HEAD_DIM;
    const float* Kb = Kc + (size_t)b * max_seq * rowStride + (size_t)kvh * HEAD_DIM;
    const float* Vb = Vc + (size_t)b * max_seq * rowStride + (size_t)kvh * HEAD_DIM;

    for (int t = w; t < n; t += N_WARPS) {
        // TODO 4: load this lane's part of the K row and the V row ONCE into registers,
        //         then score and update every query head of the group against them.
        //         Every lane must take part in every shuffle.
        const float* krow = Kb + t * rowStride;
        float partial = 0.0f;
        for (int i = 0; i < D_PER_LANE; ++i) {
            partial += qr[i] * krow[lane + i * WARP_SIZE];
        }
        const float s = warpSumAll(partial) * scale;

        const float mNew = fmaxf(m, s);
        const float alpha = expf(m - mNew);
        const float p = expf(s - mNew);
        D = D * alpha + p;
        const float* vrow = Vb + t * rowStride;
        for (int i = 0; i < D_PER_LANE; ++i) {
            O[i] = O[i] * alpha + p * vrow[lane + i * WARP_SIZE];
        }
        m = mNew;
    }

    // TODO 5: merge per query head. For each head of the group: publish the warps' state,
    //         wait, combine and write that head's output row, and wait again before the
    //         next head reuses the shared buffer.
    if (lane == 0) {
        m_s[w] = m;
        D_s[w] = D;
    }
    for (int i = 0; i < D_PER_LANE; ++i) {
        O_s[w][lane + i * WARP_SIZE] = O[i];
    }
    __syncthreads();

    float mAll = M_INIT;
    for (int x = 0; x < N_WARPS; ++x) {
        mAll = fmaxf(mAll, m_s[x]);
    }
    float weight[N_WARPS];
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
        orow[dd] = o / DAll;
    }
}

void launchDecodeAttentionGrouped(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                                  int batch, int h_q, int h_kv, int max_seq, float scale) {
    // TODO 6: a grid of (h_kv, batch) blocks of BLOCK_SIZE threads (lab 20.4 launched
    //         (h_q, batch)), and the instance for this group size, one case per g_q from
    //         1 to MAX_GQ, for example
    //             case 4: decodeAttentionGroupedKernel<4><<<grid, BLOCK_SIZE>>>(...); break;
    //         Print a message and launch nothing for a g_q above MAX_GQ.
}
