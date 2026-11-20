// Lab X1.1: top-k by radix select (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5, the top-k stage: the k largest logits of every row, without sorting the row.
// README.md has the contract, the design and the commands. Lab X1.3 calls a copy of the
// reference solution of this file.
//
// Not in the book, and ours: sampling. The parts are the book's: chapter 9's shared-memory
// histogram with atomics (Fig. 9.10, pp. 213-214), the most-significant-digit idea of section
// 14.9 (p. 346), the warp vote and popcount rank of section 12.3 (Fig. 12.3, pp. 292-293)
// inside section 12.5's stable filter (pp. 297-298), chapter 11's warp scan (Fig. 11.8, p. 264).
//
// Contract, short form: logits is [rows x cols] float32, row-major; every logit finite. For each
// row r, write kk = min(k, cols) pairs (column, logit) to outIdx[r * k + j] and outVal[r * k + j],
// j < kk: the kk largest by lab 14.3's key (orderedKey below), ties at the kk-th value to the
// lowest columns, in ascending column order. Write nothing else. scratch: 4 * rows unsigned ints.
#include <cuda/atomic>

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define DIGIT_BITS 8                            // bits per pass: 4 passes over a 32-bit key
#define BINS (1u << DIGIT_BITS)                 // histogram bins
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots in shared memory
#define SCRATCH_PER_ROW 4                       // what topkThreshold leaves for topkGather, per row

// ---------------------------------------------------------------- provided: lab 12.4's helpers
// The vote's type and the bit counts that match it. CUDA: 32-bit. HIP: every _sync intrinsic
// static_asserts a 64-bit mask and __ballot_sync returns 64 bits (infra-facts B3); __popc or
// __ffs on such a mask would silently see only its low 32 lanes (ch12-facts B1).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
__device__ unsigned int popcMask(lane_mask_t m) { return __popcll(m); }
__device__ unsigned int firstLane(lane_mask_t m) { return __ffsll((long long)m) - 1; }   // m != 0
#else
typedef unsigned int lane_mask_t;
__device__ unsigned int popcMask(lane_mask_t m) { return __popc(m); }
__device__ unsigned int firstLane(lane_mask_t m) { return __ffs((int)m) - 1; }           // m != 0
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// Provided: lab 14.3's key, ascending: a larger logit has a larger key as an unsigned integer,
// and -0.0 has the same key as +0.0 (CUB's documented transform, ch14-facts A3).
__device__ unsigned int orderedKey(unsigned int u) {
    if (u == 0x80000000u) {
        u = 0u;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------- 1. the threshold
// One block per row. Find, one byte of the key at a time from the top, the kk-th largest key
// K*, and leave in scratch[row * SCRATCH_PER_ROW ...] K* and how many keys equal to K* the
// gather must keep (the tie budget).
__global__ void topkThreshold(const unsigned int* logitBits, unsigned int* scratch, int cols, int k) {
    __shared__ unsigned int hist_s[BINS];

    // TODO 1. State that every thread holds the same copy of: the bits of K* decided so far (a
    //         prefix), which bits those are (a mask), and how many keys are still to be placed
    //         (start: kk). Each pass handles one byte, most significant first.
    //
    // TODO 2. Per pass: zero the histogram; walk the row (thread t takes columns t, t + 256, ...),
    //         and for every key whose decided bits equal the prefix, add one to the bin of the
    //         current byte (a block-scope atomic, as in lab 14.3's histogram). Barriers between
    //         zeroing, counting and reading.
    //
    // TODO 3. Find the crossing bin: walking down from bin 255, the bin where the running count
    //         first reaches the keys still needed. One warp can do it, warpSize bins at a time,
    //         with a warp scan and a ballot (firstLane picks the first lane that got there).
    //         Hand the bin, and the count of the bins above it, to the whole block.
    //
    // TODO 4. Every key in a bin above is in; every key in a bin below is out. Extend the prefix
    //         and the mask, reduce what is still needed. If every key of the crossing bin is
    //         needed, the set is decided: you may stop early (then which K* and which budget
    //         make the gather keep exactly that bin? README.md, "The design"). Mind the barrier
    //         before the next pass overwrites what the threads are still reading.
    //
    // TODO 5. Thread 0 writes the row's result to scratch.
}

// ---------------------------------------------------------------- 2. the gather
// One block per row, one pass over the row in rounds of blockDim.x columns. Write each kept
// (column, logit bits) pair to its slot; slots are in column order.
__global__ void topkGather(const unsigned int* logitBits, const unsigned int* scratch, int* outIdx,
                           unsigned int* outBits, int cols, int k) {
    __shared__ unsigned int warpTies_s[MAX_WARPS];
    __shared__ unsigned int warpKeeps_s[MAX_WARPS];

    // TODO 6. Classify the key at column base + threadIdx.x: above K*, equal to it (a tie), or out.
    //
    // TODO 7. A tie is kept while the number of ties before it in the row, in column order, is
    //         below the budget topkThreshold left. Rank it as lab 12.4 ranks survivors: a ballot,
    //         a popcount of the lanes below, warp totals in shared memory, counts carried from
    //         round to round. Then rank the kept keys the same way: that rank is the slot.
    //
    // TODO 8. Remember that the output row stride is k, and that a row has only kk pairs.
}

// ---------------------------------------------------------------- the launcher
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch, int rows, int cols,
                    int k) {
    // TODO 9. Hand the kernels the float32 tensors as 32-bit unsigned integers (a pointer cast, as
    //         lab 14.3 does: no kernel ever needs a float operation), then launch topkThreshold and
    //         topkGather, one block of BLOCK_SIZE threads per row.
}
