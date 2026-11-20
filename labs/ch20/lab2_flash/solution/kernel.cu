// Lab 20.2 reference solution (CUDA; also built for HIP and for the CPU emulators).
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
// Where this file differs from the page, and why (notes/today/ch20-facts.md unless named):
//   - One head of one sequence on the page; here grid y walks batch x heads, the "additional
//     dimensions of thread blocks" p. 495 mentions, and each block offsets its pointers.
//   - Any n. The page divides N by B_r and B_c, so N must be a multiple of both (notes,
//     "Unclear"). Here T_r and T_c round up, load_Q and load_KT_and_V fill rows past n with
//     zeros, the mask also hides columns >= n, and store_O skips rows >= n.
//   - The tiles are declared in the kernel and passed to the helpers. Metal forbids
//     threadgroup arrays in helper functions (C2) and AMD's documentation is silent (B3).
//   - cub::WarpReduce with one TempStorage shared by all 16 warps (Figs. 20.12, 20.15) is not
//     CUB's documented use (A2), and cub::Max() and cub::Sum() are gone (infra A11). Here a
//     xor butterfly of shuffles leaves the max or the sum in every lane, so the book's
//     broadcast from lane 0 is not needed.
//   - __shfl_sync(0xFFFFFFFF, ...) does not compile for HIP (B1): the mask has its own type,
//     and every shuffle passes width WARP_SIZE = 32 (B2, see below).
//   - A __syncwarp() after the P write in compute_P_and_update_D. In compute_O every lane
//     reads the whole P row, written by the other lanes; shuffles do not order memory, so
//     without it nothing guarantees the reads see the writes (A3; llama.cpp has the same
//     barrier, D1). Lab 20.3 is about this line.
//   - NEG_INFINITY is not a CUDA name (A5). Nothing here is infinite: a masked score is never
//     used (both functions test the mask by index), the identity of max is -FLT_MAX, and the
//     running max starts at -FLT_MAX/2, as llama.cpp's does, so that no subtraction of two
//     huge values can overflow (A5, D1). Exercise 3: O_i and D_i start at 0 (FA2's
//     Algorithm 1, A7).
//   - out_D -> L = m + log(D), the logsumexp FlashAttention-2 stores (A7); lab 20.1 writes
//     the same.
//   - Tiles wholly above the diagonal are skipped, as FlashAttention-2 does (A7). The page
//     runs all T_c tiles for every panel; the masked ones only cost time.
//   - B_c is TILE_WIDTH from the build: 32 as on the page, 16 also built. Each lane owns at
//     most one column of a tile (B_c <= 32), and a lane with no column still takes part in
//     every shuffle.
//   - The page's dd loop in compute_S_and_max is split into a loop over the D_SIZE register
//     slots and one over the 32 source lanes. Same products, same order; the register slot
//     is then a counter of a 4-step loop rather than dd / 32. Whether Q_i and O_i stay in
//     registers is the compiler's decision (unverified on every backend).
//   - The page's d, a macro, is HEAD_DIM here (a macro named d would rewrite every d in the
//     file), and d_size is D_SIZE. exp and log are written expf and logf.
//   - compute_P_and_update_D returns nothing: the page returns curr_sum_warp and never uses it.
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
#define B_c TILE_WIDTH                     // columns of K^T (rows of V) per tile
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
// warpSize (64), would read across the two halves (ch20-facts B2). So every shuffle below
// passes width WARP_SIZE, which keeps each exchange inside its own 32 lanes. The mask names
// every lane of the hardware warp: both halves run the same shuffles. HIP wants that mask
// as a 64-bit integer (infra-facts B3).
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
// Tuned for 32 banks: AMD has 32 or 64 (B3), Apple documents none.
static __device__ __forceinline__ int addr(int x) { return x + (x >> LOG_NUM_BANKS); }

static __device__ __forceinline__ float maxOp(float a, float b) { return a > b ? a : b; }

// ---------------------------------------------------------------- provided helpers
// Fig. 20.10: this warp's B_r_warp rows of panel i of Q, interleaved: lane L holds columns
// L, L + 32, L + 64, L + 96. Rows past n are zeros.
static __device__ void load_Q(const float* Q, float Q_i[B_r_warp][D_SIZE], int i, int n) {
    for (int ii = 0; ii < B_r_warp; ii++) {
        const int row = B_r * i + B_r_warp * warpIdx() + ii;
        for (int dd = laneIdx(), ddd = 0; dd < HEAD_DIM; dd += WARP_SIZE, ddd++) {
            Q_i[ii][ddd] = row < n ? Q[(size_t)row * HEAD_DIM + dd] : 0.0f;
        }
    }
}

// Fig. 20.11: tile j of K, stored transposed (HEAD_DIM rows of B_c columns, padded through
// addr), and tile j of V. Rows of K and V past n are zeros. As on the page, only threads 0
// to HEAD_DIM - 1 load (exercise 4 asks why every load is coalesced).
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

// ---------------------------------------------------------------- the reader's part
// Exercise 3: the running output, denominator and max of this thread's rows before any tile.
static __device__ void initialize(float O_i[B_r_warp][D_SIZE], float* D_i, float* m_i) {
    for (int ii = 0; ii < B_r_warp; ii++) {
        for (int ddd = 0; ddd < D_SIZE; ddd++) {
            O_i[ii][ddd] = 0.0f;    // nothing accumulated yet
        }
        D_i[ii] = 0.0f;             // an empty sum
        m_i[ii] = -FLT_MAX / 2;     // below every real score, and finite (A5)
    }
}

// Fig. 20.12: row ii of this warp against tile j of K^T. Lane jj computes S[row][col] for
// col = B_c * j + jj, scales it, and stores it in S_i. Returns, to every lane, the largest
// visible score of the segment (-FLT_MAX if none is visible).
static __device__ float compute_S_and_max(const float Q_i[B_r_warp][D_SIZE], const float* KT_j,
                                          float (*S_i)[B_c], int i, int j, int ii, int n, float scale) {
    const int jj = laneIdx();
    const bool hasCol = jj < B_c;                 // false for lanes 16 to 31 when B_c = 16
    float S_ij = 0.0f;
    for (int ddd = 0; ddd < D_SIZE; ddd++) {
        for (int src = 0; src < WARP_SIZE; src++) {
            // Element dd = 32 * ddd + src of this Q row lives in lane src, slot ddd. Every
            // lane takes part in the shuffle, a lane with no column too.
            const float q = __shfl_sync(FULL_MASK, Q_i[ii][ddd], src, WARP_SIZE);
            if (hasCol) {
                S_ij += q * KT_j[addr((ddd * WARP_SIZE + src) * B_c + jj)];
            }
        }
    }
    const int row = B_r * i + B_r_warp * warpIdx() + ii;
    const int col = B_c * j + jj;
    float curr_max = -FLT_MAX;
    if (hasCol) {
        S_ij = scale * S_ij;
        S_i[B_r_warp * warpIdx() + ii][jj] = S_ij;
        if (col <= row && col < n) {              // the causal mask, by index
            curr_max = S_ij;
        }
    }
    // Butterfly: after log2(32) steps every lane holds the max of all 32.
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        curr_max = maxOp(curr_max, __shfl_xor_sync(FULL_MASK, curr_max, offset, WARP_SIZE));
    }
    return curr_max;
}

// Fig. 20.14, as on the page: fold the segment's max into the running max, rescale the
// running denominator to it, and return the OLD max (compute_O needs it).
static __device__ float update_m_and_D(float* m_i, float* D_i, float curr_max_warp, int ii) {
    const float last_m = m_i[ii];
    float D = D_i[ii];
    if (m_i[ii] < curr_max_warp) {
        m_i[ii] = curr_max_warp;
        D *= expf(last_m - m_i[ii]);
    }
    D_i[ii] = D;
    return last_m;
}

// Fig. 20.15: turn the S segment into the P segment in place, exp(S - m) with the updated
// max (0 where masked), and add its sum to the running denominator.
static __device__ void compute_P_and_update_D(float (*S_i)[B_c], const float* m_i, float* D_i, int i, int j, int ii,
                                              int n) {
    const int jj = laneIdx();
    const int row = B_r * i + B_r_warp * warpIdx() + ii;
    const int col = B_c * j + jj;
    float P_ij = 0.0f;
    if (jj < B_c) {
        if (col <= row && col < n) {
            P_ij = expf(S_i[B_r_warp * warpIdx() + ii][jj] - m_i[ii]);
        }
        S_i[B_r_warp * warpIdx() + ii][jj] = P_ij;
    }
    float sum = P_ij;
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset, WARP_SIZE);
    }
    D_i[ii] += sum;
    // compute_O reads the whole P row, which the other lanes of this warp just wrote. The
    // shuffles above do not order memory; this barrier does (A3). The page has none.
    __syncwarp();
}

// Fig. 20.13: rescale this thread's part of the O row for the new max, then add the P
// segment times the V tile. Lane L owns the O columns L, L + 32, L + 64, L + 96.
static __device__ void compute_O(float (*S_i)[B_c], float (*V_j)[HEAD_DIM], float O_i[B_r_warp][D_SIZE], int ii,
                                 float last_m, const float* m_i) {
    const float* P_row = S_i[B_r_warp * warpIdx() + ii];
    const float alpha = expf(last_m - m_i[ii]);   // 1 when the max did not change
    for (int dd = laneIdx(), ddd = 0; dd < HEAD_DIM; dd += WARP_SIZE, ddd++) {
        O_i[ii][ddd] *= alpha;                     // rescale first ...
        float O_ij = 0.0f;
        for (int jj = 0; jj < B_c; jj++) {
            O_ij += P_row[jj] * V_j[jj][dd];       // ... then accumulate the new tile
        }
        O_i[ii][ddd] += O_ij;
    }
}

// Fig. 20.9. Block (x, y) handles panels x, x + gridDim.x, ... of head y.
__global__ void flashAttentionKernel(const float* Q, const float* K, const float* V, float* O, float* L, int n,
                                     float scale) {
    // The tiles, declared here and passed down (Metal requires it, C2).
    __shared__ float KT_j[KT_SIZE];
    __shared__ float S_i[B_r][B_c];
    __shared__ float V_j[B_c][HEAD_DIM];

    // This block's (sequence, head).
    const size_t head = blockIdx.y;
    const float* Qh = Q + head * n * HEAD_DIM;
    const float* Kh = K + head * n * HEAD_DIM;
    const float* Vh = V + head * n * HEAD_DIM;
    float* Oh = O + head * n * HEAD_DIM;
    float* Lh = L + head * n;

    const int T_r = (n + B_r - 1) / B_r;   // panels of Q and O (the page's N / B_r, rounded up)
    const int T_c = (n + B_c - 1) / B_c;   // tiles of K^T and V

    for (int i = blockIdx.x; i < T_r; i += gridDim.x) {
        float O_i[B_r_warp][D_SIZE];
        float D_i[B_r_warp];
        float m_i[B_r_warp];
        initialize(O_i, D_i, m_i);

        float Q_i[B_r_warp][D_SIZE];
        load_Q(Qh, Q_i, i, n);

        // The panel's last row is B_r * (i + 1) - 1: tiles that start after it are all masked.
        const int lastTile = (B_r * (i + 1) - 1) / B_c + 1;
        const int tiles = lastTile < T_c ? lastTile : T_c;
        for (int j = 0; j < tiles; j++) {
            load_KT_and_V(Kh, Vh, KT_j, V_j, j, n);
            __syncthreads();   // the tiles are complete before anyone reads them

            for (int ii = 0; ii < B_r_warp; ii++) {
                const float curr_max_warp = compute_S_and_max(Q_i, KT_j, S_i, i, j, ii, n, scale);
                const float last_m = update_m_and_D(m_i, D_i, curr_max_warp, ii);
                compute_P_and_update_D(S_i, m_i, D_i, i, j, ii, n);
                compute_O(S_i, V_j, O_i, ii, last_m, m_i);
            }
            __syncthreads();   // every warp is done with the tiles before the next load replaces them
        }
        store_O(Oh, Lh, O_i, D_i, m_i, i, n);
    }
}

// Exercise 1: the host side. One block per panel of Q (the kernel's i loop also allows a
// smaller grid), one per (sequence, head): grid y is at most 65535 on CUDA (ch05-facts A5).
void launchFlashAttention(const float* Q, const float* K, const float* V, float* O, float* L, int batch, int heads,
                          int n, float scale) {
    if (batch <= 0 || heads <= 0 || n <= 0) return;
    dim3 grid((n + B_r - 1) / B_r, batch * heads);
    flashAttentionKernel<<<grid, BLOCK_SIZE>>>(Q, K, V, O, L, n, scale);
}
