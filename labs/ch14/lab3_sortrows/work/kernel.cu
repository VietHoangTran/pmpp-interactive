// Lab 14.3 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5 (sorted candidates): every row of a [rows x cols] float32 tensor sorted in
// descending order, stably, as column indices and as values.
//
// Book: radix sort as a chain of stable partitions (section 14.4, pp. 334-336), the
// destination formulas of Fig. 14.6 (p. 338), the kernel of Fig. 14.7 (p. 339), the local
// sort in shared memory (14.6), a digit of r bits as r one-bit local passes (14.7), several
// keys per thread (14.8). The book prints no code for 14.6 to 14.8; they are exercises 1 to 3
// (p. 347). Read README.md first: it states the contract, the key and the design.
//
// Contract:
//   probs is [rows x cols] float32, row-major, batch first; rows >= 1, cols >= 1 (any value);
//   any floats, including -0.0, infinities, subnormals and NaN. For every row r:
//     sortedIdx[r][p]   = the column holding the p-th largest value; equal values by
//                         INCREASING column (a stable sort)
//     sortedProbs[r][p] = probs[r][sortedIdx[r][p]], the same 32 bits
//   scratch holds 4 * rows * cols unsigned ints for you: keys and indices, twice.
//   The order is that of the key in README.md, "The contract": CUB's transform for floats.
//
// The design (README.md, "The design"): one block of BLOCK_SIZE threads per row; three kernels.
#include <cuda/atomic>

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define RADIX_BITS 8                            // r: your choice; 8 is suggested (32 / r passes)
#define RADIX (1u << RADIX_BITS)                // 2^r buckets
#define KEYS_PER_THREAD 4                       // c: your choice; a tile is BLOCK_SIZE * c keys
#define TILE (BLOCK_SIZE * KEYS_PER_THREAD)
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots for the block scan

// From lab 10.1: the shuffle mask has its own type (HIP requires 64 bits), and the warp width
// is warpSize, never a literal 32.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

// ---------------------------------------------------------------- provided: block scan
// Chapter 11's scan-scan-add on unsigned counts: the warp scan of Fig. 11.8, then the warp
// totals. Returns the sum of `val` over the threads before this one in threadIdx.x order
// (exclusive), and the block's total in *total. Every thread of the block must call it: it
// holds a barrier and full-mask shuffles. warpSums_s has MAX_WARPS entries. Two calls in a
// row need a barrier between them: the second writes warpSums_s while the first may still be
// reading it. (Unlike Fig. 11.9, every thread adds up the warp totals itself.)
__device__ unsigned int blockExclusiveScan(unsigned int val, unsigned int* warpSums_s,
                                           unsigned int* total) {
    const lane_mask_t full = ~(lane_mask_t)0;
    const unsigned int warps = blockDim.x / warpSize;
    unsigned int incl = val;
    for (unsigned int stride = 1; stride < (unsigned int)warpSize; stride *= 2) {
        unsigned int left = __shfl_up_sync(full, incl, stride);
        if (laneIdx() >= stride) {
            incl += left;
        }
    }
    if (laneIdx() == (unsigned int)warpSize - 1) {
        warpSums_s[warpIdx()] = incl;
    }
    __syncthreads();
    unsigned int before = 0, sum = 0;
    for (unsigned int w = 0; w < warps; ++w) {
        const unsigned int s = warpSums_s[w];
        before += w < warpIdx() ? s : 0u;
        sum += s;
    }
    *total = sum;
    return incl - val + before;
}

// ---------------------------------------------------------------- 1. keys
__global__ void descKeys(const unsigned int* probsBits, unsigned int* keys, unsigned int* idx,
                         int cols) {
    // TODO 1: block blockIdx.x owns row blockIdx.x. For each of its columns c (a loop strided
    //         by blockDim.x), read the 32 bits of the value, build the key that README.md
    //         describes step by step, and store it with c as its index. Integer operations
    //         only: probsBits is the float tensor seen as unsigned, and the kernel should
    //         never turn it back into floats (README.md, "Why bits, not floats").
    //         Check your key by hand on -1.0, -0.0, +0.0 and 1.0 before you run anything.
}

// ---------------------------------------------------------------- 2. one digit
// One stable pass over the digit at bits [shift, shift + RADIX_BITS) of every key of a row,
// from (keysIn, idxIn) to (keysOut, idxOut). Block b owns row b.
__global__ void sortRowsPass(const unsigned int* keysIn, const unsigned int* idxIn,
                             unsigned int* keysOut, unsigned int* idxOut, int cols,
                             unsigned int shift) {
    // TODO 2: count. A histogram of the row's digits in shared memory, RADIX bins, built
    //         with chapter 9's block-scope atomics (cuda::atomic_ref, Fig. 9.10). Zero it,
    //         count, and make sure nobody reads a bin before every thread has counted.
    //
    // TODO 3: scan. blockExclusiveScan of the counts (one bucket per thread) gives where each
    //         bucket starts in the output row. That start is the bucket's running offset: the
    //         next free slot for that digit.
    //
    // TODO 4: the tiles, in order, TILE keys at a time (the last one may be short).
    //   a. Load the tile's keys and indices into shared memory, coalesced.
    //   b. Sort the tile by the digit with RADIX_BITS stable one-bit splits, low bit first
    //      (section 14.7), each with Fig. 14.6's two formulas. Each thread owns
    //      KEYS_PER_THREAD keys (section 14.8). The scan counts in threadIdx.x order; which
    //      positions must a thread own for that to be the order of the positions?
    //   c. Write each key and index to keysOut and idxOut: its bucket's running offset plus
    //      its rank among this tile's keys of the same digit. The tile is sorted by digit
    //      now: where does each digit's run begin? Arrange the writes so that consecutive
    //      threads write consecutive slots (section 14.6).
    //   d. Advance every running offset by the number of this tile's keys with its digit.
    // For each shared array ask: who writes it, who reads it next, and what orders the two?
    // A barrier must sit between every such pair, including between the last use in one tile
    // and the first in the next.
}

// ---------------------------------------------------------------- 3. gather
// sortedBits is sortedProbs seen as unsigned.
__global__ void gatherRows(const unsigned int* probsBits, const unsigned int* idxSorted,
                           unsigned int* sortedBits, int* sortedIdx, int cols) {
    // TODO 5: for each position p of row blockIdx.x: the column idxSorted holds there goes
    //         to sortedIdx, and the 32 bits of probs at that column to sortedBits.
}

// ---------------------------------------------------------------- the sort
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx,
                        unsigned int* scratch, int rows, int cols) {
    // TODO 6: hand the kernels probs and sortedProbs as unsigned int pointers (a cast).
    //         Split scratch into two key buffers and two index buffers of rows * cols
    //         entries each. Launch descKeys, then one sortRowsPass per digit, least
    //         significant digit first, each reading the buffers the previous one wrote; then
    //         gatherRows from whichever index buffer holds the last pass's result. One block
    //         of BLOCK_SIZE threads per row for every kernel.
}
