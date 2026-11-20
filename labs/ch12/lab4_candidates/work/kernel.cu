// Lab 12.4 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5 piece: candidate compaction for top-k, top-p and min-p sampling.
//
// Book: the stable filter of section 12.5 (Fig. 12.7, Fig. 12.8, pp. 297-298): a keep flag
// per key, an exclusive scan of the flags, a write at the offset; and the warp vote and
// binary prefix sum of section 12.3 (Fig. 12.3, p. 293). Carrying (column, value) pairs and
// one block per row are not in the book. Read README.md first: it states the contract.
//
// Contract:
//   logits is [rows x cols] float32, row-major, batch first; threshold[rows]; rows >= 1,
//   cols >= 1 (any value, not only multiples of anything); every value finite. For row r,
//   the columns c with logits[r][c] >= threshold[r], in ASCENDING order, are written as
//     outIdx[r * cols + k] = c,  outVal[r * cols + k] = logits[r][c]   for k < counts[r]
//   and counts[r] is how many there are. Nothing else in outIdx and outVal may be written.
//   One block of BLOCK_SIZE threads per row. No global atomics, and no block may wait for
//   another (README: why).

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // enough per-warp slots on every backend

// From lab 10.1: the vote's mask has its own type (HIP requires 64 bits), and the warp width
// is warpSize, never a literal 32.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// TODO 1: a popcount for lane_mask_t. On CUDA the mask is 32 bits. On HIP it is 64 bits,
//         and a wavefront may have 64 lanes: which popcount counts all of them, and what
//         would the other one do, silently? (README: Portability.)

__global__ void compactCandidatesKernel(const float* logits, const float* threshold, int* outIdx,
                                        float* outVal, unsigned int* counts, int cols) {
    // TODO 2: block blockIdx.x owns row blockIdx.x. Walk the row in rounds of blockDim.x
    //         columns: in each round thread t looks at column base + t (coalesced), and
    //         decides keep = (that logit >= this row's threshold). Columns past the end of
    //         the row are not kept, but their threads must still reach every vote and
    //         every barrier the others reach.
    //
    // TODO 3: a kept pair's position in the output is
    //             (kept in earlier rounds) + (kept by earlier warps this round)
    //                                      + (kept by lower lanes of my warp this round).
    //         - Lower lanes of my warp: one vote of the whole warp, then the binary prefix
    //           sum of Fig. 12.3, lines 22-24. Fig. 12.3 votes with __activemask() inside
    //           the `if`; why does that not give you this lane's position here?
    //         - Earlier warps: each warp publishes its total in shared memory; after a
    //           barrier, each thread can add up the totals before its own warp.
    //         - Earlier rounds: every thread can keep the same running count. Where does it
    //           need to live, and does it need an atomic?
    //         Then ask what the NEXT round writes to shared memory, and who may still be
    //         reading the totals of this one.
    //
    // TODO 4: after the last round, one thread writes counts[blockIdx.x].
}

void launchCompactCandidates(const float* logits, const float* threshold, int* outIdx, float* outVal,
                             unsigned int* counts, int rows, int cols) {
    // TODO 5: one block of BLOCK_SIZE threads per row.
}
