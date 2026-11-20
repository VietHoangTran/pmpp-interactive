// Lab 12.4 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5 piece: candidate compaction for top-k, top-p and min-p sampling.
//
// Book: the stable filter of section 12.5 (Fig. 12.7, Fig. 12.8, pp. 297-298): a keep flag
// per key, an exclusive scan of the flags, a write at the offset. Inside a warp the scan is
// the binary prefix sum of section 12.3 (Fig. 12.3, lines 22-24, p. 293): a vote, a mask of
// the lanes below, a popcount. Carrying (column, value) pairs and one block per row are ours
// (notes/ch12.md, "Relevance to LLM-inference kernels").
//
// Contract (README.md states it in full):
//   logits is [rows x cols] float32, row-major, batch first; threshold[rows]; rows >= 1,
//   cols >= 1 (any value); every logit and threshold finite. For row r, the columns c with
//   logits[r][c] >= threshold[r], in ascending order, are written as
//     outIdx[r * cols + k] = c,  outVal[r * cols + k] = logits[r][c]   for k < counts[r]
//   and counts[r] is how many there are. Nothing else in outIdx and outVal is written.
//   One block of BLOCK_SIZE threads per row; no global atomics; no block waits for another.
//
// Where this file differs from the page, and why:
//   - Fig. 12.8 gets each offset from gridExclusiveScan, a grid-wide scan in which blocks
//     wait for earlier blocks; the book never shows it. Here one block owns a row and walks
//     it in rounds of blockDim.x columns, carrying the row's running count from round to
//     round itself. No block waits for another: Apple documents no progress guarantee
//     between threadgroups (notes/today/ch12-facts.md C4), and AMD none between workgroups
//     (B6), so this is the design that is documented-safe on every backend here.
//   - Fig. 12.3 calls __activemask() inside the `if`. NVIDIA documents that it does not tell
//     which lanes took a branch (A1), and a stable filter needs every lane's position, kept
//     or not. So every lane votes with __ballot_sync in uniform control flow.
//   - Fig. 12.3 writes (1 << laneIdx()) - 1 and __popc on an unsigned int mask. HIP's masks
//     are 64 bits and its wavefronts may have 64 lanes (B1): the mask has its own type and
//     the popcount is chosen to match it, as llama.cpp does (D1).
//   - Fig. 12.8 reads input[i] with no i < N guard. Here a thread past the end of the row
//     reads nothing but still votes and reaches both barriers, with keep = false.

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has (ch10-facts B2)
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots in shared memory

// The vote's type and its popcount. CUDA: 32-bit, __popc. HIP: every _sync intrinsic
// static_asserts a 64-bit mask and __ballot_sync returns 64 bits (infra-facts B3); __popc
// would silently count only the low 32 lanes of a 64-wide wavefront (ch12-facts B1).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
__device__ unsigned int popcMask(lane_mask_t m) { return __popcll(m); }
#else
typedef unsigned int lane_mask_t;
__device__ unsigned int popcMask(lane_mask_t m) { return __popc(m); }
#endif

// The book's helpers (Chapter 10, pp. 238-239), with warpSize in place of the constant 32.
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

__global__ void compactCandidatesKernel(const float* logits, const float* threshold, int* outIdx,
                                        float* outVal, unsigned int* counts, int cols) {
    __shared__ unsigned int warpTotals_s[MAX_WARPS];   // survivors per warp, this round

    const size_t rowStart = (size_t)blockIdx.x * cols;  // block b owns row b
    const float* in = logits + rowStart;
    int* idxOut = outIdx + rowStart;
    float* valOut = outVal + rowStart;
    const float thr = threshold[blockIdx.x];

    const lane_mask_t full = ~(lane_mask_t)0;
    const lane_mask_t lanesBelow = ((lane_mask_t)1 << laneIdx()) - 1;   // bits 0 .. lane-1

    // Survivors in the rounds already done. Every thread computes the same value, so it
    // lives in a register and needs no shared memory and no atomic.
    unsigned int rowCount = 0;

    for (int base = 0; base < cols; base += blockDim.x) {   // the same trip count in every thread
        // Keep flag (Fig. 12.7). The block reads one contiguous run per round: coalesced.
        const int c = base + threadIdx.x;
        float val = 0.0f;
        bool keep = false;
        if (c < cols) {
            val = in[c];
            keep = val >= thr;
        }

        // Rank among this warp's survivors: the binary prefix sum of section 12.3, over a
        // vote that every lane takes part in.
        const lane_mask_t vote = __ballot_sync(full, keep);
        const unsigned int rankInWarp = popcMask(vote & lanesBelow);
        if (laneIdx() == 0) {
            warpTotals_s[warpIdx()] = popcMask(vote);
        }
        __syncthreads();   // every warp's total is in shared memory

        // Exclusive scan of the warp totals: each thread adds up the warps before its own,
        // and the whole round. At most MAX_WARPS values, the same addresses for every thread.
        unsigned int before = 0, roundTotal = 0;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            const unsigned int n = warpTotals_s[w];
            before += w < warpIdx() ? n : 0;
            roundTotal += n;
        }
        __syncthreads();   // every thread has read the totals before the next round overwrites them

        // The write at the offset (Fig. 12.8, lines 07-09), after everything kept earlier.
        if (keep) {
            const unsigned int k = rowCount + before + rankInWarp;
            idxOut[k] = c;
            valOut[k] = val;
        }
        rowCount += roundTotal;
    }

    if (threadIdx.x == 0) {
        counts[blockIdx.x] = rowCount;
    }
}

void launchCompactCandidates(const float* logits, const float* threshold, int* outIdx, float* outVal,
                             unsigned int* counts, int rows, int cols) {
    compactCandidatesKernel<<<rows, BLOCK_SIZE>>>(logits, threshold, outIdx, outVal, counts, cols);
}
