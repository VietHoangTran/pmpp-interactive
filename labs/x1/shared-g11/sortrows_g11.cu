// PROVIDED, do not edit. A copy of labs/ch14/lab3_sortrows/solution/kernel.cu (lab 14.3's
// row-wise stable descending sort), linked into labs in module X1 beside other copies.
// The only change: laneIdx and warpIdx are static, so that no two linked files define them.
// app/src/chapters/x1/copies-g11.test.ts checks that nothing else differs.
// Lab 14.3 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5 (sorted candidates): every row of a [rows x cols] float32 tensor sorted in
// descending order, stably, as column indices and as values.
//
// Book: radix sort as a sequence of stable partitions (section 14.4, pp. 334-336), the
// destination formulas of Fig. 14.6 (p. 338) and the kernel of Fig. 14.7 (p. 339); the local
// sort in shared memory before the global write (14.6, pp. 339-342); a multi-bit digit done
// as r one-bit local splits (14.7, pp. 342-344); several keys per thread (14.8, pp. 344-345).
// The book prints no code for 14.6 to 14.8: they are exercises 1 to 3 (p. 347). The shared
// histogram is chapter 9's (Fig. 9.10, pp. 213-214) and the block scan chapter 11's
// (Figs. 11.8 and 11.9, p. 264).
// Ours, not the book's: the float-to-key transform (CUB documents the same one,
// notes/today/ch14-facts.md A3), one block per row, the running offset per bucket, the gather.
//
// Contract (README.md states it in full):
//   probs is [rows x cols] float32, row-major, batch first; rows >= 1, cols >= 1; any values,
//   negatives, both zeros, infinities, subnormals and NaN included. For every row r:
//     sortedIdx[r][p]   = the column holding the p-th largest value; equal values by
//                         increasing column (a stable sort)
//     sortedProbs[r][p] = probs[r][sortedIdx[r][p]], the same 32 bits
//   scratch holds 4 * rows * cols unsigned ints: keys and indices, twice (ping-pong).
//   The order is that of the key built in descKeys: descending numeric order, -0.0 equal to
//   +0.0, and NaN by its bits after the transform (+NaN before everything, -NaN after
//   everything), which is CUB's documented rule (ch14-facts A3).
//
// Where this file differs from Fig. 14.7, and why:
//   - Fig. 14.7 calls gridExclusiveScan, which the book never defines, from inside the kernel
//     (notes/ch14.md, "Unclear in the PDF", item 6). Here one block owns a whole row, so every
//     scan is a block scan and no block waits for another.
//   - Fig. 14.7 reads the number of ones at bits[N], one entry past the N the figure shows
//     (same list, item 4). blockExclusiveScan below returns the total separately.
//   - Fig. 14.7 moves bare unsigned keys by one bit per launch. Here one launch handles a
//     RADIX_BITS-bit digit (exercise 2), sorts each tile in shared memory before writing it
//     out (exercise 1) with KEYS_PER_THREAD keys per thread (exercise 3), and carries each
//     key's column with it (the key-value sort of p. 330).
//   - Fig. 14.9's table of bucket sizes per block is not needed: one block visits the tiles of
//     its row in order, so a running offset per bucket does the job of the scanned table.
#include <cuda/atomic>

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define RADIX_BITS 8                            // r: bits per digit, so 32 / r passes
#define RADIX (1u << RADIX_BITS)                // 2^r buckets
#define KEYS_PER_THREAD 8                       // c: consecutive keys each thread owns in a tile
#define TILE (BLOCK_SIZE * KEYS_PER_THREAD)     // keys sorted in shared memory at a time
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots for the block scan

static_assert(32 % RADIX_BITS == 0, "the passes must cover the 32 key bits exactly");
static_assert(RADIX <= BLOCK_SIZE, "the bucket scan gives each thread at most one bucket");

// The mask of __shfl_up_sync. CUDA: unsigned. HIP: every _sync intrinsic static_asserts a
// 64-bit mask (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
static __device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
static __device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

// ---------------------------------------------------------------- provided: block scan
// Chapter 11's scan-scan-add on unsigned counts: the warp scan of Fig. 11.8, then the warp
// totals. Returns the sum of `val` over the threads before this one (exclusive), and the
// block's total in *total. Every thread of the block must call it: it holds a barrier and
// full-mask shuffles. warpSums_s has MAX_WARPS entries. Two calls in a row need a barrier
// between them: the second writes warpSums_s while the first may still be reading it.
// One difference from Fig. 11.9 (ours): a block of 256 has at most 8 warp totals, so every
// thread adds up the ones it needs instead of warp 0 scanning them. That saves a barrier
// and a shuffle loop per call, and this lab calls the scan once per key bit for every tile,
// 32 times per tile over the whole sort. (Not measured
// on CUDA or HIP here. On Metal a loop like this one was slower than a SIMD-group prefix
// sum, which is why kernel.metal does stage 2 differently.)
__device__ unsigned int blockExclusiveScan(unsigned int val, unsigned int* warpSums_s,
                                           unsigned int* total) {
    const lane_mask_t full = ~(lane_mask_t)0;
    const unsigned int warps = blockDim.x / warpSize;

    // Stage 1: every warp scans its own values (inclusive), in registers.
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

    // Stage 2: the totals of the warps before this one, and of all warps.
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
// probsBits is the float32 tensor seen as 32-bit unsigned integers: the launcher casts the
// pointer and this kernel never touches a float, so none of the liberties a math mode may
// take with floats applies (for Metal: no signed zeros, NaN and INF undefined, subnormals
// flushed in arithmetic, ch14-facts C3).
__global__ void descKeys(const unsigned int* probsBits, unsigned int* keys, unsigned int* idx,
                         int cols) {
    const size_t row = (size_t)blockIdx.x * (size_t)cols;
    for (unsigned int c = threadIdx.x; c < (unsigned int)cols; c += blockDim.x) {
        unsigned int u = probsBits[row + c];
        if (u == 0x80000000u) {
            u = 0u;              // -0.0 becomes +0.0: equal keys, so the sort keeps their order
        }
        // Unsigned order = numeric order: a non-negative float gets its sign bit set (it
        // ranks above every negative); a negative one has every bit inverted (the larger
        // its magnitude, the smaller its key).
        const unsigned int key = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
        keys[row + c] = ~key;    // descending: invert, then sort ascending (ties keep order)
        idx[row + c] = c;
    }
}

// ---------------------------------------------------------------- 2. one digit
__device__ __forceinline__ unsigned int digitOf(unsigned int key, unsigned int shift) {
    return (key >> shift) & (RADIX - 1u);
}

// One stable pass over the digit at bits [shift, shift + RADIX_BITS) of every key of a row.
// Block b owns row b.
__global__ void sortRowsPass(const unsigned int* keysIn, const unsigned int* idxIn,
                             unsigned int* keysOut, unsigned int* idxOut, int cols,
                             unsigned int shift) {
    __shared__ unsigned int offset_s[RADIX];      // the histogram, then each bucket's next slot
    __shared__ unsigned int tileStart_s[RADIX];   // where each bucket starts in the sorted tile
    __shared__ unsigned int keys_s[TILE];
    __shared__ unsigned int idx_s[TILE];
    __shared__ unsigned int warpSums_s[MAX_WARPS];

    const unsigned int n = (unsigned int)cols;
    const size_t row = (size_t)blockIdx.x * n;
    const unsigned int* keysRow = keysIn + row;
    const unsigned int* idxRow = idxIn + row;

    // (a) Count the row's digits: chapter 9's private histogram in shared memory.
    for (unsigned int d = threadIdx.x; d < RADIX; d += blockDim.x) {
        offset_s[d] = 0u;
    }
    __syncthreads();
    for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin(offset_s[digitOf(keysRow[c], shift)]);
        bin.fetch_add(1u, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // (b) Exclusive scan of the counts: where each bucket starts in the output row. Thread d
    //     reads and then overwrites bucket d only.
    unsigned int total = 0;
    const unsigned int count = threadIdx.x < RADIX ? offset_s[threadIdx.x] : 0u;
    const unsigned int start = blockExclusiveScan(count, warpSums_s, &total);
    if (threadIdx.x < RADIX) {
        offset_s[threadIdx.x] = start;
    }
    // No barrier here: the one after the first tile's load orders this write before any read.

    // (c) The tiles of the row, in order.
    const unsigned int first = threadIdx.x * KEYS_PER_THREAD;   // this thread's positions
    for (unsigned int base = 0; base < n; base += TILE) {
        const unsigned int m = n - base < TILE ? n - base : TILE;   // keys in this tile

        // Load: consecutive threads read consecutive keys (coalesced).
        for (unsigned int j = threadIdx.x; j < m; j += blockDim.x) {
            keys_s[j] = keysRow[base + j];
            idx_s[j] = idxRow[base + j];
        }
        __syncthreads();

        // Sort the tile by the digit: RADIX_BITS stable one-bit splits, low bit first (14.7).
        // Thread t owns positions first .. first + KEYS_PER_THREAD - 1 (14.8), so the scan
        // order of the threads is the order of the positions.
        for (unsigned int b = 0; b < RADIX_BITS; ++b) {
            const unsigned int bitPos = shift + b;
            unsigned int k[KEYS_PER_THREAD];
            unsigned int v[KEYS_PER_THREAD];
            unsigned int ones = 0;
            for (unsigned int i = 0; i < KEYS_PER_THREAD; ++i) {
                k[i] = 0u;
                v[i] = 0u;
                if (first + i < m) {
                    k[i] = keys_s[first + i];
                    v[i] = idx_s[first + i];
                    ones += (k[i] >> bitPos) & 1u;
                }
            }
            unsigned int onesTotal = 0;
            unsigned int onesBefore = blockExclusiveScan(ones, warpSums_s, &onesTotal);
            // The scan's barriers also mean that every thread has read its keys, so the tile
            // may now be overwritten.
            for (unsigned int i = 0; i < KEYS_PER_THREAD; ++i) {
                const unsigned int p = first + i;
                if (p < m) {
                    const unsigned int bit = (k[i] >> bitPos) & 1u;
                    // Fig. 14.6: a zero goes to (its index - ones before it), a one to
                    // (keys - ones in total + ones before it).
                    const unsigned int dst = bit == 0u ? p - onesBefore : m - onesTotal + onesBefore;
                    keys_s[dst] = k[i];
                    idx_s[dst] = v[i];
                    onesBefore += bit;
                }
            }
            __syncthreads();
        }

        // Where each bucket starts in the sorted tile: wherever the digit changes.
        for (unsigned int p = threadIdx.x; p < m; p += blockDim.x) {
            const unsigned int d = digitOf(keys_s[p], shift);
            if (p == 0 || digitOf(keys_s[p - 1], shift) != d) {
                tileStart_s[d] = p;
            }
        }
        __syncthreads();

        // Each key goes to its bucket's next free slots. Consecutive threads write consecutive
        // positions of a bucket: the coalesced write of section 14.6. Earlier tiles' keys of the
        // same bucket are already below offset_s[d], so the pass is stable across tiles too.
        for (unsigned int p = threadIdx.x; p < m; p += blockDim.x) {
            const unsigned int d = digitOf(keys_s[p], shift);
            const size_t dst = row + offset_s[d] + (p - tileStart_s[d]);
            keysOut[dst] = keys_s[p];
            idxOut[dst] = idx_s[p];
        }
        __syncthreads();   // every thread has read offset_s

        // Advance the running offsets: the last key of each bucket adds the bucket's size.
        for (unsigned int p = threadIdx.x; p < m; p += blockDim.x) {
            const unsigned int d = digitOf(keys_s[p], shift);
            if (p == m - 1 || digitOf(keys_s[p + 1], shift) != d) {
                offset_s[d] += p + 1 - tileStart_s[d];
            }
        }
        __syncthreads();   // before the next tile reads offset_s and overwrites the tile
    }
}

// ---------------------------------------------------------------- 3. gather
// sortedBits is sortedProbs seen as unsigned: the 32 bits are copied, never converted.
__global__ void gatherRows(const unsigned int* probsBits, const unsigned int* idxSorted,
                           unsigned int* sortedBits, int* sortedIdx, int cols) {
    const size_t row = (size_t)blockIdx.x * (size_t)cols;
    for (unsigned int p = threadIdx.x; p < (unsigned int)cols; p += blockDim.x) {
        const unsigned int c = idxSorted[row + p];
        sortedIdx[row + p] = (int)c;
        sortedBits[row + p] = probsBits[row + c];
    }
}

// ---------------------------------------------------------------- the sort
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx,
                        unsigned int* scratch, int rows, int cols) {
    if (rows <= 0 || cols <= 0) return;
    const size_t n = (size_t)rows * (size_t)cols;
    // The kernels see both float tensors as 32-bit unsigned integers (README: why).
    const unsigned int* probsBits = (const unsigned int*)probs;
    unsigned int* sortedBits = (unsigned int*)sortedProbs;
    // scratch: keys A | indices A | keys B | indices B, n entries each.
    unsigned int* keys[2] = {scratch, scratch + 2 * n};
    unsigned int* idx[2] = {scratch + n, scratch + 3 * n};

    descKeys<<<rows, BLOCK_SIZE>>>(probsBits, keys[0], idx[0], cols);
    int cur = 0;
    for (unsigned int shift = 0; shift < 32; shift += RADIX_BITS) {   // least significant digit first
        sortRowsPass<<<rows, BLOCK_SIZE>>>(keys[cur], idx[cur], keys[1 - cur], idx[1 - cur], cols, shift);
        cur = 1 - cur;
    }
    gatherRows<<<rows, BLOCK_SIZE>>>(probsBits, idx[cur], sortedBits, sortedIdx, cols);
}
