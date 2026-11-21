// Lab 20.5 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Grouped-query decode: lab 20.4's decode attention, regrouped so that each loaded K and V
// row serves every query head that reads it.
//
// Book: section 20.7 (pp. 508-509) describes MQA and GQA and their arithmetic intensity,
// Eq. 20.13: about g_q, the number of query heads per K/V head (notes/today/ch20-facts.md E3
// for that meaning of g_q). The book prints no kernel for it; this one is ours, and does
// what llama.cpp's gqa_ratio path does: several Q heads against one loaded K/V head (D1).
//
// Contract: lab 20.4's launchDecodeAttention exactly (README.md), plus g_q = h_q / h_kv <= MAX_GQ.
//   q, out [batch, h_q, HEAD_DIM]; Kc, Vc [batch, max_seq, h_kv, HEAD_DIM]; len [batch],
//   1 <= len[b] <= max_seq; query head h reads K/V head h / g_q.
//
// Design: one block of 256 threads per (K/V head, sequence), serving query heads
// kvh * g_q ... kvh * g_q + g_q - 1. Warp w still takes keys w, w + 8, ...; for each key it
// loads the K row and the V row once and updates g_q running states (m, D, O~), one per
// query head. At the end the warps merge per query head through shared memory (Eqs. 20.4
// and 20.6), one head at a time, so the shared output buffer stays HEAD_DIM floats per warp:
// all eight heads at once would be 32 KiB, Metal's whole threadgroup limit (ch20-facts C1).
//
// The group size is a template parameter, GQ, and the launcher picks the instance. Every
// array below then has a size and every loop a trip count the compiler knows, so the arrays
// can live in registers and hold exactly g_q states. (A first version sized every array for
// the largest group and switched the extra states off with `g < g_q` at run time. Its Metal
// form measured slower than lab 20.4's per-head kernel even at g_q = 1, where the two do
// the same work, on the one Apple GPU it was written on; sized for one head it matched.)
//
// Everything lab 20.4's solution says about the page holds here too: finite M_INIT (A5),
// a logical 32-lane warp with width 32 on every shuffle (B2), a typed lane mask (B1).
#include <cfloat>
#include <cstdio>

#define HEAD_DIM 128
#define WARP_SIZE 32                           // the book's WARP_SIZE: a logical warp of 32 lanes
#define BLOCK_SIZE 256
#define N_WARPS (BLOCK_SIZE / WARP_SIZE)
#define D_PER_LANE (HEAD_DIM / WARP_SIZE)
#define M_INIT (-FLT_MAX / 2.0f)
#define MAX_GQ 8                               // the largest group the launcher has an instance for

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ int laneIdx() { return threadIdx.x % WARP_SIZE; }
__device__ int warpIdx() { return threadIdx.x / WARP_SIZE; }

// Block (kvh, b): K/V head kvh of sequence b and its GQ query heads.
template <int GQ>
__global__ void decodeAttentionGroupedKernel(const float* q, const float* Kc, const float* Vc, const int* len,
                                             float* out, int h_q, int h_kv, int max_seq, float scale) {
    __shared__ float m_s[N_WARPS][GQ];
    __shared__ float D_s[N_WARPS][GQ];
    __shared__ float O_s[N_WARPS][HEAD_DIM];       // one query head's O~ per warp, reused per head

    const int kvh = blockIdx.x;
    const int b = blockIdx.y;
    const int h0 = kvh * GQ;                       // first query head of the group
    const int lane = laneIdx();
    const int w = warpIdx();
    const int n = len[b];
    const lane_mask_t full = ~(lane_mask_t)0;

    // The group's queries and running states, in registers.
    float qr[GQ][D_PER_LANE];
    float m[GQ], D[GQ], O[GQ][D_PER_LANE];
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
        const float* qrow = q + ((size_t)b * h_q + h0 + g) * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            qr[g][i] = qrow[lane + i * WARP_SIZE];
            O[g][i] = 0.0f;
        }
        m[g] = M_INIT;
        D[g] = 0.0f;
    }

    const size_t rowStride = (size_t)h_kv * HEAD_DIM;
    const float* Kb = Kc + (size_t)b * max_seq * rowStride + (size_t)kvh * HEAD_DIM;
    const float* Vb = Vc + (size_t)b * max_seq * rowStride + (size_t)kvh * HEAD_DIM;

    for (int t = w; t < n; t += N_WARPS) {
        // One load of the K row and the V row, for every query head of the group.
        float kr[D_PER_LANE], vr[D_PER_LANE];
        const float* krow = Kb + t * rowStride;
        const float* vrow = Vb + t * rowStride;
#pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            kr[i] = krow[lane + i * WARP_SIZE];
            vr[i] = vrow[lane + i * WARP_SIZE];
        }

        // GQ partial dot products, then GQ butterflies side by side.
        float s[GQ];
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
            s[g] = 0.0f;
#pragma unroll
            for (int i = 0; i < D_PER_LANE; ++i) {
                s[g] += qr[g][i] * kr[i];
            }
        }
#pragma unroll
        for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
#pragma unroll
            for (int g = 0; g < GQ; ++g) {
                s[g] += __shfl_xor_sync(full, s[g], offset, WARP_SIZE);
            }
        }

        // The online update of lab 20.4, once per query head.
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
            const float sg = s[g] * scale;
            const float mNew = fmaxf(m[g], sg);
            const float alpha = expf(m[g] - mNew);
            const float p = expf(sg - mNew);
            D[g] = D[g] * alpha + p;
#pragma unroll
            for (int i = 0; i < D_PER_LANE; ++i) {
                O[g][i] = O[g][i] * alpha + p * vr[i];
            }
            m[g] = mNew;
        }
    }

    // Merge, per query head. Every warp's (m, D) for all heads goes out at once; O~ goes
    // one head at a time through the same buffer.
    if (lane == 0) {
#pragma unroll
        for (int g = 0; g < GQ; ++g) {
            m_s[w][g] = m[g];
            D_s[w][g] = D[g];
        }
    }
#pragma unroll
    for (int g = 0; g < GQ; ++g) {
#pragma unroll
        for (int i = 0; i < D_PER_LANE; ++i) {
            O_s[w][lane + i * WARP_SIZE] = O[g][i];
        }
        __syncthreads();                           // head g's states, and every m and D, are visible

        float mAll = M_INIT;
        for (int x = 0; x < N_WARPS; ++x) {
            mAll = fmaxf(mAll, m_s[x][g]);
        }
        float weight[N_WARPS];
        float DAll = 0.0f;
        for (int x = 0; x < N_WARPS; ++x) {
            weight[x] = expf(m_s[x][g] - mAll);
            DAll += D_s[x][g] * weight[x];
        }
        float* orow = out + ((size_t)b * h_q + h0 + g) * HEAD_DIM;
        for (int dd = threadIdx.x; dd < HEAD_DIM; dd += blockDim.x) {
            float o = 0.0f;
            for (int x = 0; x < N_WARPS; ++x) {
                o += O_s[x][dd] * weight[x];
            }
            orow[dd] = o / DAll;
        }
        __syncthreads();                           // everyone has read O_s before head g + 1 overwrites it
    }
}

void launchDecodeAttentionGrouped(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                                  int batch, int h_q, int h_kv, int max_seq, float scale) {
    if (batch <= 0 || h_kv <= 0) {
        return;
    }
    dim3 dimGrid(h_kv, batch);                     // one block per (K/V head, sequence)
    switch (h_q / h_kv) {                          // one instance per group size
        case 1: decodeAttentionGroupedKernel<1><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 2: decodeAttentionGroupedKernel<2><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 3: decodeAttentionGroupedKernel<3><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 4: decodeAttentionGroupedKernel<4><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 5: decodeAttentionGroupedKernel<5><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 6: decodeAttentionGroupedKernel<6><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 7: decodeAttentionGroupedKernel<7><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        case 8: decodeAttentionGroupedKernel<8><<<dimGrid, BLOCK_SIZE>>>(q, Kc, Vc, len, out, h_q, h_kv, max_seq, scale); break;
        default:
            std::fprintf(stderr, "launchDecodeAttentionGrouped: g_q = %d is above MAX_GQ = %d; nothing launched\n",
                         h_q / h_kv, MAX_GQ);
    }
}
