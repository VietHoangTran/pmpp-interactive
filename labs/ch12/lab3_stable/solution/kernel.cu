// Lab 12.3 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Two stable, out-of-place filters of unsigned int keys: keep a key when key < threshold, and
// keep the kept keys in input order. Neither kernel waits for another block, so both run on
// every backend.
//
// Contract: n >= 1. With no keys there is no last key to write *outputSize, and the launchers
// would launch zero blocks, which HIP rejects (notes/today/ch02-facts.md B4; CUDA's documentation
// does not state the case). The driver never calls with n = 0.
//
// Book: the stable filter of section 12.5 (Fig. 12.7, p. 297; Fig. 12.8, p. 298), the
// shared-memory gathering and coarsening of section 12.6 (Figs. 12.9 and 12.10, pp. 298-300),
// and exercises 1, 3 and 4 (p. 302).
//
// Part A, exercise 1: three steps, each its own kernel launch.
//   keepFlagsKernel  keep[i] = input[i] < threshold
//   (provided)       offset = exclusive scan of keep          launchExclusiveScanU32
//   scatterKernel    output[offset[i]] = input[i] where keep[i]; *outputSize = offset[n-1] + keep[n-1]
//
// Part B, exercises 3 and 4 in reduce-then-scan form (the design of PyTorch's nonzero_static,
// notes/today/ch12-facts.md D4): one tile of BLOCK_DIM * coarse keys per block.
//   countKeptKernel      blockCounts[b] = how many keys of tile b are kept
//   (provided)           blockOffsets = exclusive scan of blockCounts
//   stagedScatterKernel  each block gathers its kept keys, in order, in shared memory, then
//                        writes them as one contiguous run at output[blockOffsets[b]]
//
// Where this file differs from the page, and why:
//   - Fig. 12.8 calls gridExclusiveScan, which the book never defines, inside one kernel. A
//     scan inside one kernel needs blocks to wait for earlier blocks (chapter 11's single-pass
//     scan), which only CUDA documents as safe (ch12-facts A10, B6, C4). Exercise 1's three
//     kernels need no waiting, so that is the form here.
//   - Fig. 12.8 reads input[i] with no i < n check. Every access here is guarded.
//   - The book counts a warp's passing lanes with __activemask() inside the branch (Fig. 12.3).
//     NVIDIA now warns that __activemask() "cannot be used to determine which warp lanes
//     execute a given branch" (ch12-facts A1), so the vote here is __ballot_sync over the whole
//     warp, in uniform control flow, where the result is exactly the set of kept lanes.
//   - Lane masks are 64 bits under HIP, and __popc would drop the upper half of one, so the
//     mask has its own type and its own popcount (ch12-facts B1).
#define BLOCK_DIM 256                           // threads per block, every kernel in this file
#define MIN_WARP_SIZE 32                        // the narrowest warp of any backend here
#define MAX_WARPS (BLOCK_DIM / MIN_WARP_SIZE)   // per-warp slots in shared memory

// Provided in ../shared-g12/scan_u32.cu.
void launchExclusiveScanU32(const unsigned int* in, unsigned int* out, unsigned int n, unsigned int* scratch);

// The lane mask and its popcount. CUDA: 32 bits and __popc. HIP: every _sync intrinsic takes a
// 64-bit mask and __ballot_sync returns one, which __popc would truncate (ch12-facts B1).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif
static __device__ __forceinline__ int maskPopc(lane_mask_t m) {
#if defined(__HIPCC__)
    return __popcll(m);
#else
    return __popc(m);
#endif
}
// Any other argument type (an unsigned int holding a HIP ballot, say) is a compile error.
template <typename T> static __device__ __forceinline__ int maskPopc(T m) = delete;

// The book's helpers (chapter 10), with warpSize in place of the constant 32.
__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ __forceinline__ unsigned int warpsInBlock() { return blockDim.x / warpSize; }

// ======================================================================== part A (exercise 1)

__global__ void keepFlagsKernel(const unsigned int* input, unsigned int* keep, unsigned int n,
                                unsigned int threshold) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        keep[i] = input[i] < threshold ? 1u : 0u;
    }
}

__global__ void scatterKernel(const unsigned int* input, const unsigned int* keep, const unsigned int* offset,
                              unsigned int* output, unsigned int n, unsigned int* outputSize) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (keep[i]) {
            output[offset[i]] = input[i];   // offset[i] kept keys come before this one
        }
        if (i == n - 1) {
            *outputSize = offset[i] + keep[i];   // the inclusive value of the last element
        }
    }
}

void launchStableFilterFlags(const unsigned int* input, unsigned int* output, unsigned int n,
                             unsigned int threshold, unsigned int* outputSize, unsigned int* keep,
                             unsigned int* offset, unsigned int* scanScratch) {
    const unsigned int blocks = (n + BLOCK_DIM - 1) / BLOCK_DIM;
    keepFlagsKernel<<<blocks, BLOCK_DIM>>>(input, keep, n, threshold);
    launchExclusiveScanU32(keep, offset, n, scanScratch);
    scatterKernel<<<blocks, BLOCK_DIM>>>(input, keep, offset, output, n, outputSize);
}

// ======================================================================== part B (exercises 3, 4)
// Block b owns the tile of keys [b * BLOCK_DIM * coarse, (b + 1) * BLOCK_DIM * coarse). The block
// walks it in `coarse` rounds of BLOCK_DIM consecutive keys, thread t taking key t of the round,
// so every round's loads are coalesced.

__global__ void countKeptKernel(const unsigned int* input, unsigned int n, unsigned int threshold,
                                unsigned int* blockCounts, unsigned int coarse) {
    __shared__ unsigned int warpTotals_s[MAX_WARPS];
    const lane_mask_t fullMask = ~(lane_mask_t)0;
    const unsigned int tileStart = blockIdx.x * BLOCK_DIM * coarse;

    unsigned int warpCount = 0;   // the same in every lane of a warp
    for (unsigned int r = 0; r < coarse; ++r) {
        const unsigned int i = tileStart + r * BLOCK_DIM + threadIdx.x;
        const bool keep = i < n && input[i] < threshold;
        // Every lane of the warp votes, kept or not, so the mask is exactly the kept lanes.
        warpCount += maskPopc(__ballot_sync(fullMask, keep));
    }
    if (laneIdx() == 0) {
        warpTotals_s[warpIdx()] = warpCount;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned int total = 0;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            total += warpTotals_s[w];
        }
        blockCounts[blockIdx.x] = total;
    }
}

__global__ void stagedScatterKernel(const unsigned int* input, unsigned int* output, unsigned int n,
                                    unsigned int threshold, const unsigned int* blockOffsets,
                                    unsigned int* outputSize, unsigned int coarse) {
    extern __shared__ unsigned int staged_s[];   // BLOCK_DIM * coarse keys: the block's kept keys, in order
    __shared__ unsigned int warpTotals_s[MAX_WARPS];
    const lane_mask_t fullMask = ~(lane_mask_t)0;
    const lane_mask_t lanesBelow = ((lane_mask_t)1 << laneIdx()) - 1;
    const unsigned int tileStart = blockIdx.x * BLOCK_DIM * coarse;

    unsigned int staged = 0;   // kept keys staged so far; the same in every thread
    for (unsigned int r = 0; r < coarse; ++r) {
        const unsigned int i = tileStart + r * BLOCK_DIM + threadIdx.x;
        const unsigned int key = i < n ? input[i] : 0u;
        const bool keep = i < n && key < threshold;

        // Rank inside the warp: the kept lanes below this one (the binary prefix sum of p. 293).
        const lane_mask_t kept = __ballot_sync(fullMask, keep);
        const unsigned int rankInWarp = maskPopc(kept & lanesBelow);
        if (laneIdx() == 0) {
            warpTotals_s[warpIdx()] = maskPopc(kept);
        }
        __syncthreads();   // every warp's total of this round is in shared memory

        // Rank inside the round: add the totals of the warps before this one.
        unsigned int warpsBefore = 0, roundTotal = 0;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            const unsigned int t = warpTotals_s[w];
            warpsBefore += w < warpIdx() ? t : 0u;
            roundTotal += t;
        }
        if (keep) {
            staged_s[staged + warpsBefore + rankInWarp] = key;   // appended after the earlier rounds
        }
        staged += roundTotal;
        __syncthreads();   // nobody overwrites warpTotals_s until everyone has read this round's
    }

    // The staged run is complete (the loop's last barrier). Consecutive threads write
    // consecutive keys: one contiguous run per block (Figs. 12.9 and 12.10).
    const unsigned int base = blockOffsets[blockIdx.x];
    for (unsigned int j = threadIdx.x; j < staged; j += BLOCK_DIM) {
        output[base + j] = staged_s[j];
    }
    if (blockIdx.x == gridDim.x - 1 && threadIdx.x == 0) {
        *outputSize = base + staged;   // the last block's offset plus its own count
    }
}

void launchStableFilterStaged(const unsigned int* input, unsigned int* output, unsigned int n,
                              unsigned int threshold, unsigned int* outputSize, unsigned int* blockCounts,
                              unsigned int* blockOffsets, unsigned int* scanScratch, int coarse) {
    const unsigned int tile = BLOCK_DIM * (unsigned int)coarse;
    const unsigned int blocks = (n + tile - 1) / tile;
    const size_t stagingBytes = (size_t)tile * sizeof(unsigned int);   // 16 KiB at coarse 16
    countKeptKernel<<<blocks, BLOCK_DIM>>>(input, n, threshold, blockCounts, (unsigned int)coarse);
    launchExclusiveScanU32(blockCounts, blockOffsets, blocks, scanScratch);
    stagedScatterKernel<<<blocks, BLOCK_DIM, stagingBytes>>>(input, output, n, threshold, blockOffsets,
                                                            outputSize, (unsigned int)coarse);
}
