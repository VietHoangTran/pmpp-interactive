// Lab 20.4 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M7: append one token's K and V to the KV cache, then attend one new query per
// (sequence, head) over the cache. Book: section 20.4 (pp. 488-492, Fig. 20.6) for what a
// decode step computes; section 20.5 (Eqs. 20.3 to 20.6, Figs. 20.10 to 20.16) for the
// online softmax you build it from. The book prints no decode kernel: the design below is
// ours (README.md explains it).
//
// Contract:
//   k_new, v_new  [batch, h_kv, HEAD_DIM]           q, out  [batch, h_q, HEAD_DIM]
//   Kc, Vc        [batch, max_seq, h_kv, HEAD_DIM]
//   pos, len      [batch] int, 0 <= pos[b] < max_seq, 1 <= len[b] <= max_seq
//   h_q a multiple of h_kv; query head h reads K/V head h / (h_q / h_kv).
//   Row-major float32, batch first, raw device pointers.
#include <cfloat>

// The course's CUDA front-end check declares no CUDA Math API functions, so it is given
// these two device declarations. nvcc, hipcc and the emulators never see them.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
__device__ float fmaxf(float x, float y);
#endif

#define HEAD_DIM 128                           // the book's d (Fig. 20.9)
#define WARP_SIZE 32                           // the book's WARP_SIZE, used as a LOGICAL warp of 32 lanes:
                                               // pass it as the width of every shuffle, so that on a
                                               // 64-wide AMD wavefront two logical warps never mix
#define BLOCK_SIZE 256                         // threads per (sequence, head)
#define N_WARPS (BLOCK_SIZE / WARP_SIZE)       // 8 warps per block
#define D_PER_LANE (HEAD_DIM / WARP_SIZE)      // 4 dimensions per lane
#define M_INIT (-FLT_MAX / 2.0f)               // start value for a running max: finite on purpose
                                               // (README, "Where this differs from the page")

// HIP's _sync intrinsics need a 64-bit mask; CUDA's take an unsigned int.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ int laneIdx() { return threadIdx.x % WARP_SIZE; }
__device__ int warpIdx() { return threadIdx.x / WARP_SIZE; }

// TODO 1: return the sum of v over the 32 lanes of the logical warp, in EVERY lane.
//         A loop of __shfl_xor_sync calls (mask ~(lane_mask_t)0, width WARP_SIZE) does it.
__device__ float warpSumAll(float v) {
    return v;
}

// ---------------------------------------------------------------- KV append
// TODO 2: one thread per element of k_new. Element e of sequence b's new row
//         (b * h_kv * HEAD_DIM + e in k_new) goes to row pos[b] of sequence b in Kc;
//         the same for v_new and Vc. Write nothing else, and nothing for a pos[b]
//         outside [0, max_seq).
__global__ void kvAppendKernel(const float* k_new, const float* v_new, float* Kc, float* Vc,
                               const int* pos, int batch, int max_seq, int h_kv) {
}

void launchKvAppend(const float* k_new, const float* v_new, float* Kc, float* Vc,
                    const int* pos, int batch, int max_seq, int h_kv) {
    // TODO 3: enough blocks of BLOCK_SIZE threads for batch * h_kv * HEAD_DIM elements.
}

// ---------------------------------------------------------------- decode attention
// One block of BLOCK_SIZE threads per (query head h = blockIdx.x, sequence b = blockIdx.y).
__global__ void decodeAttentionKernel(const float* q, const float* Kc, const float* Vc, const int* len,
                                      float* out, int h_q, int h_kv, int max_seq, float scale) {
    // TODO 4: the shared memory the warps merge through: each warp's m, D and its
    //         HEAD_DIM-long O~.

    // TODO 5: which K/V head does query head h read? Load this lane's D_PER_LANE elements
    //         of q into registers, Fig. 20.10's way: lane L takes dims L, L + 32, L + 64, ...

    // TODO 6: this warp's running state for the empty set of keys: m, D and O~[D_PER_LANE].

    // TODO 7: warp w visits keys t = w, w + N_WARPS, ... below len[b]. For each key:
    //   a. the score s = scale * (q . k_t): your four products, then warpSumAll
    //   b. the online update for the one-key set {t} (Figs. 20.14, 20.15, 20.13):
    //      new max; rescale D and O~ by e^(old max - new max); add this key's
    //      e^(s - new max) to D and that times v_t to O~
    //   Where in Kc does row t of K/V head g of sequence b start?

    // TODO 8: merge the warps (Eqs. 20.4 and 20.6, un-normalised): publish each warp's
    //         (m, D, O~), wait for the whole block, then combine with weights
    //         e^(m_w - m_all) and divide once (Fig. 20.16). A warp that saw no key has
    //         D = 0 and must contribute nothing.
}

void launchDecodeAttention(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                           int batch, int h_q, int h_kv, int max_seq, float scale) {
    // TODO 9: a grid of (h_q, batch) blocks of BLOCK_SIZE threads.
}
