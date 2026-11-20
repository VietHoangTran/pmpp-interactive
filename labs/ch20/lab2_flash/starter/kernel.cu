// Lab 20.2 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M6, part two: causal flash attention forward, with the logsumexp.
// Book: section 20.5, Figs. 20.8 to 20.16 (pp. 492-504), and Exercises 1 and 3 (p. 511).
//
// Contract (README.md states it in full):
//   Q, K, V, O are [batch x heads x n x HEAD_DIM] float32, row-major; L is [batch x heads x n].
//   n >= 1, any value. Causal: row r sees the columns c <= r. With s = scale * Q K^T,
//   m the row's largest visible s and D = sum of exp(s - m) over the visible columns:
//     O = (sum of exp(s - m) V over the visible columns) / D,   L = m + log(D),
//   the same L as lab 20.1. One block of BLOCK_SIZE = 512 threads per panel of B_r = 32 rows
//   of Q (grid x), one per (sequence, head) (grid y). No n x n buffer exists anywhere.
//
// The provided parts (constants, load_Q, load_KT_and_V, store_O) are already corrected for
// any n, a batch of heads and HIP; README.md lists where they and the page differ. Your
// part is marked TODO. Three things the page gets wrong that are now yours to get right:
// NEG_INFINITY is not a CUDA name (use the finite values below), the CUB warp reductions
// and their functors are gone (write the shuffles), and one barrier is missing (read the
// TODO in compute_P_and_update_D).
#include <cfloat>

// The course's CUDA front-end check declares no CUDA Math API functions.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
__device__ float logf(float x);
#endif

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

// ---------------------------------------------------------------- the book's constants
// Fig. 20.9, lines 1 to 7, with the names above.
#define WARP_SIZE 32                       // the book's warp: 32 lanes, on every backend (see below)
#define BLOCK_SIZE 512                     // threads per block
#define N_WARPS (BLOCK_SIZE / WARP_SIZE)   // 16 warps
#define B_r 32                             // rows of Q per block: one panel
#define B_r_warp (B_r / N_WARPS)           // 2 rows per warp
#define B_c TILE_WIDTH                     // columns of K^T (rows of V) per tile: 32 or 16
#define HEAD_DIM 128                       // the book's d
#define D_SIZE (HEAD_DIM / WARP_SIZE)      // the book's d_size: elements of a Q or O row per lane
#define LOG_NUM_BANKS 5                    // 32 banks of 4 bytes (NVIDIA, infra-facts A10)
#define KT_SIZE (B_c * HEAD_DIM + ((B_c * HEAD_DIM) >> LOG_NUM_BANKS))   // K^T tile with its padding

#if B_c > WARP_SIZE || WARP_SIZE % B_c != 0
#error "each lane owns at most one column of a tile here: B_c (TILE_WIDTH) must be 8, 16 or 32"
#endif

// ---------------------------------------------------------------- lanes and warps
// The book's warp is 32 lanes wide. On a 64-wide AMD wavefront (gfx9) that is half a
// wavefront: two of the book's warps share one, and HIP's shuffle, whose width defaults to
// warpSize (64), would read across the two halves (ch20-facts B2). So EVERY shuffle you
// write passes width WARP_SIZE as its last argument, which keeps each exchange inside its
// own 32 lanes. Its mask is FULL_MASK: both halves run the same shuffles. HIP wants that
// mask as a 64-bit integer (infra-facts B3), hence the type.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif
#define FULL_MASK (~(lane_mask_t)0)

// The book's chapter 10 helpers (p. 498), for the 32-lane warp above.
static __device__ __forceinline__ int laneIdx() { return threadIdx.x % WARP_SIZE; }
static __device__ __forceinline__ int warpIdx() { return threadIdx.x / WARP_SIZE; }

// The padding of the K^T tile against bank conflicts (p. 499; Chapter 16 calls it pad(x)).
// Every access to KT_j goes through it.
static __device__ __forceinline__ int addr(int x) { return x + (x >> LOG_NUM_BANKS); }

static __device__ __forceinline__ float maxOp(float a, float b) { return a > b ? a : b; }

// ---------------------------------------------------------------- provided helpers
// Fig. 20.10: this warp's B_r_warp rows of panel i of Q, interleaved: lane L holds columns
// L, L + 32, L + 64, L + 96 in Q_i[ii][0..3]. Rows past n are zeros.
static __device__ void load_Q(const float* Q, float Q_i[B_r_warp][D_SIZE], int i, int n) {
    for (int ii = 0; ii < B_r_warp; ii++) {
        const int row = B_r * i + B_r_warp * warpIdx() + ii;
        for (int dd = laneIdx(), ddd = 0; dd < HEAD_DIM; dd += WARP_SIZE, ddd++) {
            Q_i[ii][ddd] = row < n ? Q[(size_t)row * HEAD_DIM + dd] : 0.0f;
        }
    }
}

// Fig. 20.11: tile j of K, stored transposed (HEAD_DIM rows of B_c columns, padded through
// addr: element (dd, jj) is KT_j[addr(dd * B_c + jj)]), and tile j of V (V_j[jj][dd]).
// Rows of K and V past n are zeros. Called by every thread of the block; as on the page,
// only threads 0 to HEAD_DIM - 1 load.
static __device__ void load_KT_and_V(const float* K, const float* V, float* KT_j, float (*V_j)[HEAD_DIM], int j,
                                     int n) {
    for (int jj = 0; jj < B_c; jj++) {
        const int col = B_c * j + jj;   // the token whose key and value this is
        for (int dd = threadIdx.x; dd < HEAD_DIM; dd += blockDim.x) {
            KT_j[addr(dd * B_c + jj)] = col < n ? K[(size_t)col * HEAD_DIM + dd] : 0.0f;
            V_j[jj][dd] = col < n ? V[(size_t)col * HEAD_DIM + dd] : 0.0f;
        }
    }
}

// Fig. 20.16: O = O~ / D for this warp's rows, and L = m + log(D) where the page writes D.
// Rows past n are not stored.
static __device__ void store_O(float* O, float* L, const float O_i[B_r_warp][D_SIZE], const float* D_i,
                               const float* m_i, int i, int n) {
    for (int ii = 0; ii < B_r_warp; ii++) {
        const int row = B_r * i + B_r_warp * warpIdx() + ii;
        if (row >= n) continue;
        for (int ddd = 0; ddd < D_SIZE; ddd++) {
            const int col = ddd * WARP_SIZE + laneIdx();
            O[(size_t)row * HEAD_DIM + col] = O_i[ii][ddd] / D_i[ii];
        }
        if (laneIdx() == 0) {
            L[row] = m_i[ii] + logf(D_i[ii]);
        }
    }
}

// ---------------------------------------------------------------- your part
// Exercise 3: the running output, denominator and max of this thread's rows before any tile.
static __device__ void initialize(float O_i[B_r_warp][D_SIZE], float* D_i, float* m_i) {
    // TODO 1: set every O_i and D_i to the value an empty sum has, and every m_i to a value
    //         below every real score. Not -infinity: compute_O computes exp(last_m - m_i),
    //         and if both were -infinity that is exp(NaN) (ch20-facts A5). A large finite
    //         value whose difference with any score cannot overflow: -FLT_MAX / 2, as
    //         llama.cpp uses.
}

// Fig. 20.12: row ii of this warp against tile j of K^T. Lane jj computes S[row][col] for
// col = B_c * j + jj, scales it, and stores it in S_i. Returns, to EVERY lane, the largest
// visible score of the segment (-FLT_MAX if none is visible).
static __device__ float compute_S_and_max(const float Q_i[B_r_warp][D_SIZE], const float* KT_j,
                                          float (*S_i)[B_c], int i, int j, int ii, int n, float scale) {
    // TODO 2: the dot product of this warp's Q row ii with column jj = laneIdx() of the
    //         K^T tile. Element dd of the Q row lives in lane dd % 32, slot dd / 32: fetch it
    //         with __shfl_sync (FULL_MASK, width WARP_SIZE). Every lane must take part in
    //         every shuffle, including a lane with no column when B_c < 32.
    // TODO 3: scale it, store it in this warp's row of S_i (which row of S_i is that?), and
    //         keep it as this lane's candidate for the max only if the column is visible:
    //         col <= row and col < n. A masked score must never become the max.
    // TODO 4: the max over the 32 lanes, in every lane: a butterfly of __shfl_xor_sync
    //         (the page's WarpReduce gives it to lane 0 only, then broadcasts).
    return -FLT_MAX;
}

// Fig. 20.14: fold the segment's max into the running max, rescale the running
// denominator to it, and return the max compute_O will need.
static __device__ float update_m_and_D(float* m_i, float* D_i, float curr_max_warp, int ii) {
    // TODO 5: Eq. 20.4's first term. Which max does compute_O need to rescale O: the one
    //         before this tile or the one after? Return that one.
    return m_i[ii];
}

// Fig. 20.15: turn the S segment into the P segment in place, exp(S - m) with the updated
// max (exactly 0 where masked), and add the segment's sum to the running denominator.
static __device__ void compute_P_and_update_D(float (*S_i)[B_c], const float* m_i, float* D_i, int i, int j, int ii,
                                              int n) {
    // TODO 6: P in place of S for this lane's column, 0 for a masked column (test the mask
    //         by index again: never pass a masked score to expf).
    // TODO 7: the sum over the 32 lanes, in every lane (a butterfly again), added to D_i[ii].
    // TODO 8: compute_O will read the WHOLE P row of this warp, which the other lanes just
    //         wrote. Shuffles do not order memory (ch20-facts A3). What must come here? The
    //         page has nothing.
}

// Fig. 20.13: rescale this thread's part of the O row for the new max, then add the P
// segment times the V tile. Lane L owns the O columns L, L + 32, L + 64, L + 96.
static __device__ void compute_O(float (*S_i)[B_c], float (*V_j)[HEAD_DIM], float O_i[B_r_warp][D_SIZE], int ii,
                                 float last_m, const float* m_i) {
    // TODO 9: for each of this lane's 4 columns: multiply O_i by exp(last_m - m_i[ii]), then
    //         add the sum over jj < B_c of P[jj] * V_j[jj][column]. In that order: why?
}

// Fig. 20.9. Block (x, y) handles panels x, x + gridDim.x, ... of head y.
__global__ void flashAttentionKernel(const float* Q, const float* K, const float* V, float* O, float* L, int n,
                                     float scale) {
    // The tiles, declared here and passed down (Metal requires it, ch20-facts C2).
    __shared__ float KT_j[KT_SIZE];
    __shared__ float S_i[B_r][B_c];
    __shared__ float V_j[B_c][HEAD_DIM];

    // This block's (sequence, head): grid y walks batch x heads.
    const size_t head = blockIdx.y;
    const float* Qh = Q + head * n * HEAD_DIM;
    const float* Kh = K + head * n * HEAD_DIM;
    const float* Vh = V + head * n * HEAD_DIM;
    float* Oh = O + head * n * HEAD_DIM;
    float* Lh = L + head * n;

    const int T_r = (n + B_r - 1) / B_r;   // panels of Q and O (the page's N / B_r, rounded up)
    const int T_c = (n + B_c - 1) / B_c;   // tiles of K^T and V

    // TODO 10: the page's outer loop over panels i (starting at blockIdx.x, strided by
    //          gridDim.x). For each panel: per-thread O_i, D_i, m_i (initialize them) and
    //          Q_i (load_Q); then the loop over tiles j: load_KT_and_V, a barrier, the warp
    //          loop over ii (compute_S_and_max, update_m_and_D, compute_P_and_update_D,
    //          compute_O), and a second barrier (what would go wrong without it?); finally
    //          store_O.
    //          Optional: tiles that start after the panel's last row are wholly masked. You
    //          may stop the j loop before them, as FlashAttention-2 does.
}

// Exercise 1: the host side.
void launchFlashAttention(const float* Q, const float* K, const float* V, float* O, float* L, int batch, int heads,
                          int n, float scale) {
    // TODO 11: a grid with one block per panel of B_r rows along x and one per (sequence,
    //          head) along y (at most 65535 on CUDA, ch05-facts A5), and BLOCK_SIZE threads
    //          per block.
}
