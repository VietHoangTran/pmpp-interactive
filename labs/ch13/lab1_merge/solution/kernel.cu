// Lab 13.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Ordered merge of two sorted arrays with co-rank partitioning.
// Book: sections 13.2 to 13.6 (pp. 305-320), Figs. 13.2, 13.5 and 13.9, and the idea
// of Exercise 3 (p. 328).
//
// Where this file differs from the page (README, "Where this lab differs from the page"):
// - Elements are KV pairs compared on .key, so the tests can see stability.
// - co_rank and merge_sequential are __device__: an unqualified function is host-only
//   on CUDA and HIP. The book's listings show no qualifier.
// - The kernels produce the first `total` outputs, total <= m + n (ours, not the book's).
// - The per-thread count is an argument; the launcher sizes the grid from it with
//   integer ceiling division. Fig. 13.9 line 03 takes ceil() of an integer quotient,
//   which has already been truncated, so it rounds nothing up.
// - A thread whose first output is at or past `total` returns; the book clamps k_next only.
// - Part (b) is not Figs. 13.11-13.13. It loads exactly the elements each block will use
//   (the alternative of p. 316 and Exercise 3), one tile per block, with the block-level
//   co-ranks computed by a separate partition launch. There is no while-loop over tiles
//   and no circular buffer (section 13.7).
#include "../merge.h"

// ---------------------------------------------------------------- Part (a)
// Fig. 13.5, p. 310. Returns i: of the first k outputs of the stable merge of A and B,
// i come from A and k - i from B. Ties go to A.
__device__ int co_rank(int k, const KV* A, int m, const KV* B, int n) {
    int i = k < m ? k : m;                 // i = min(k, m)
    int j = k - i;
    int i_low = 0 > (k - n) ? 0 : k - n;   // i_low = max(0, k - n)
    int j_low = 0 > (k - m) ? 0 : k - m;   // j_low = max(0, k - m)
    int delta;
    bool active = true;
    while (active) {
        // The bounds tests come first: && does not evaluate its right operand when the
        // left one is false, so A[i-1], B[j], B[j-1] and A[i] are read only when they exist.
        if (i > 0 && j < n && A[i - 1].key > B[j].key) {          // i is too high
            delta = ((i - i_low + 1) >> 1);                         // ceil((i - i_low) / 2)
            j_low = j;
            j = j + delta;
            i = i - delta;
        } else if (j > 0 && i < m && B[j - 1].key >= A[i].key) {  // j is too high
            delta = ((j - j_low + 1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        } else {
            active = false;
        }
    }
    return i;
}

// Fig. 13.2, p. 305. The <= sends ties to A, which is what makes the merge stable.
__device__ void merge_sequential(const KV* A, int m, const KV* B, int n, KV* C) {
    int i = 0, j = 0, k = 0;
    while (i < m && j < n) {
        if (A[i].key <= B[j].key) {
            C[k++] = A[i++];
        } else {
            C[k++] = B[j++];
        }
    }
    if (i == m) {
        while (j < n) C[k++] = B[j++];
    } else {
        while (i < m) C[k++] = A[i++];
    }
}

// Fig. 13.9, p. 313. Thread tid produces C[tid*perThread .. (tid+1)*perThread), clamped to total.
__global__ void mergeBasicKernel(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int k_curr = tid * perThread;
    if (k_curr >= total) return;                      // past the end: nothing to produce
    int k_next = k_curr + perThread < total ? k_curr + perThread : total;
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next - i_curr, &B[j_curr], j_next - j_curr, &C[k_curr]);
}

void launchMergeBasic(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread) {
    if (total == 0) return;   // HIP rejects a grid of 0 blocks (ch02-facts B4); the emulator does too
    const int perBlock = MERGE_BLOCK * perThread;
    const int blocks = (total + perBlock - 1) / perBlock;   // integer ceiling division
    mergeBasicKernel<<<blocks, MERGE_BLOCK>>>(A, m, B, n, C, total, perThread);
}

// ---------------------------------------------------------------- Part (b)
// Block-level co-ranks, one per tile boundary: bounds[b] is how many of the first
// min(b * tileSize, total) outputs come from A, for b = 0 .. numTiles.
__global__ void partitionKernel(const KV* A, int m, const KV* B, int n, int total, int tileSize, int* bounds) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    int numTiles = (total + tileSize - 1) / tileSize;
    if (b <= numTiles) {
        int k = b * tileSize < total ? b * tileSize : total;
        bounds[b] = co_rank(k, A, m, B, n);
    }
}

// One tile per block. Block b produces C[C_curr .. C_next) from A[A_curr .. A_next) and
// B[B_curr .. B_next), which together hold exactly C_next - C_curr elements.
__global__ void mergeTiledKernel(const KV* A, const KV* B, KV* C, int total, int perThread, const int* bounds) {
    extern __shared__ KV tile[];                      // A's part, then B's part

    int tileSize = blockDim.x * perThread;
    int C_curr = blockIdx.x * tileSize;
    int C_next = C_curr + tileSize < total ? C_curr + tileSize : total;
    int A_curr = bounds[blockIdx.x];
    int A_next = bounds[blockIdx.x + 1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int C_length = C_next - C_curr;                   // == A_length + B_length

    KV* A_S = &tile[0];
    KV* B_S = &tile[A_length];

    // Coalesced loads: consecutive threads copy consecutive elements (Fig. 13.12's loops,
    // bounded by what this block will use instead of by a whole tile per input).
    for (int i = 0; i < A_length; i += blockDim.x) {
        if (i + threadIdx.x < A_length) A_S[i + threadIdx.x] = A[A_curr + i + threadIdx.x];
    }
    for (int i = 0; i < B_length; i += blockDim.x) {
        if (i + threadIdx.x < B_length) B_S[i + threadIdx.x] = B[B_curr + i + threadIdx.x];
    }
    __syncthreads();                                  // the tile is complete before anyone searches it

    // Fig. 13.13 lines 37-45: this thread's slice of the block's output, clamped to the
    // block's length (the last block may be short). A thread with nothing to do merges
    // zero elements; no thread returns before the barrier above.
    int c_curr = threadIdx.x * perThread;
    int c_next = c_curr + perThread;
    c_curr = c_curr < C_length ? c_curr : C_length;
    c_next = c_next < C_length ? c_next : C_length;
    int a_curr = co_rank(c_curr, A_S, A_length, B_S, B_length);
    int a_next = co_rank(c_next, A_S, A_length, B_S, B_length);
    int b_curr = c_curr - a_curr;
    int b_next = c_next - a_next;
    merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr, C + C_curr + c_curr);
}

void launchMergeTiled(const KV* A, int m, const KV* B, int n, KV* C, int total, int perThread, int* bounds) {
    if (total == 0) return;
    const int tileSize = MERGE_BLOCK * perThread;
    const int numTiles = (total + tileSize - 1) / tileSize;
    // numTiles + 1 boundaries, one thread each.
    partitionKernel<<<(numTiles + 1 + MERGE_BLOCK - 1) / MERGE_BLOCK, MERGE_BLOCK>>>(A, m, B, n, total, tileSize, bounds);
    // Same stream, so the merge starts after the partition has finished.
    mergeTiledKernel<<<numTiles, MERGE_BLOCK, tileSize * sizeof(KV)>>>(A, B, C, total, perThread, bounds);
}

// ---------------------------------------------------------------- Provided
// Lets the tests call your co_rank directly: out[k] = co_rank(k, A, m, B, n) for
// k = 0 .. m + n, one thread per k. Do not edit.
__global__ void coRankKernel(const KV* A, int m, const KV* B, int n, int* out) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k <= m + n) out[k] = co_rank(k, A, m, B, n);
}

void launchCoRank(const KV* A, int m, const KV* B, int n, int* out) {
    const int queries = m + n + 1;
    coRankKernel<<<(queries + MERGE_BLOCK - 1) / MERGE_BLOCK, MERGE_BLOCK>>>(A, m, B, n, out);
}
