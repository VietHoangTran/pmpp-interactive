// PROVIDED, do not edit. A copy of labs/x1/lab1_topk/solution/kernel.cu (lab X1.1: top-k by radix select),
// linked into lab X1.3 beside other copies. The only change: helper functions are static, so
// that no two linked files define the same name. app/src/chapters/x1/copies-g11.test.ts checks
// that nothing else differs.
// Lab X1.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5, the top-k stage: the k largest logits of every row, found by an 8-bit MSD
// radix select instead of a sort. Lab X1.3 calls a copy of this file
// (labs/x1/shared-g11/topk_g11.cu).
//
// Not in the book, and ours: sampling (PLAN.md: "Sampling is not in the book at all"). The
// parts are the book's: the shared-memory histogram with atomics of chapter 9 (Fig. 9.10,
// pp. 213-214), the most-significant-digit idea of section 14.9 (p. 346), the warp vote and
// popcount rank of section 12.3 (Fig. 12.3, pp. 292-293) inside the stable filter of section
// 12.5 (pp. 297-298), and the warp scan of chapter 11 (Fig. 11.8, p. 264). llama.cpp's CUDA
// and Metal top-k and PyTorch's torch.topk are selections of this kind (notes/today/ch14-facts.md
// D2, D5).
//
// Contract (README.md states it in full):
//   logits is [rows x cols] float32, row-major, batch first; rows >= 1, cols >= 1, k >= 1;
//   every logit finite (-0.0 and +0.0 allowed). Let kk = min(k, cols). For every row r,
//   outIdx[r * k + j] and outVal[r * k + j], j < kk, are the columns and logits of the kk
//   largest logits of the row, in ascending column order. "Largest" is the order of lab 14.3's
//   key (below), so -0.0 equals +0.0; every tie at the kk-th value goes to the lowest columns.
//   Nothing else is written: not slots kk .. k-1 of a row, not past rows * k. outVal holds the
//   32 bits of the logit, copied, never converted. scratch holds 4 * rows unsigned ints.
//
// The design (ours): one block of 256 threads per row, two kernels.
//   topkThreshold  up to four passes, one per byte of the key, most significant first. Each
//                  counts the current byte of the keys whose higher bytes equal the prefix found
//                  so far, in a 256-bin shared histogram; walks the bins from the top until the
//                  running count reaches the k still needed; that bin's byte joins the prefix.
//                  Keys in higher bins are in, keys in lower bins are out. When every key of the
//                  crossing bin is needed the set is decided and it stops early (section X1.4's
//                  stepper shows this). It writes K* and the number of keys equal to K* still
//                  needed. After four passes K* is the kk-th largest key itself. After an early
//                  stop K* is the prefix with its undecided bits zero, the smallest key the
//                  crossing bin can hold: every key of that bin is then greater than or equal to
//                  K*, and the budget is the whole bin, so the gather keeps all of it.
//   topkGather     one pass in rounds of 256 columns: above = key > K*, tie = key == K*. A tie is
//                  kept while its rank among the row's ties is below the budget; a kept key's
//                  slot is its rank among the kept. Ranks come from a ballot and a popcount inside
//                  each warp, warp totals in shared memory, and running counts carried from round
//                  to round (lab 12.4's stable filter with a tie budget).
//
// The kernels never touch a float: the launcher hands them the logits as 32-bit unsigned
// integers, as lab 14.3 does, so no math mode can rewrite a -0.0 test (ch14-facts C3).
#include <cuda/atomic>

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define DIGIT_BITS 8                            // bits per pass: 4 passes over a 32-bit key
#define BINS (1u << DIGIT_BITS)                 // histogram bins
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots in shared memory
#define SCRATCH_PER_ROW 4                       // K*, ties still needed, two unused

// ---------------------------------------------------------------- provided: lab 12.4's helpers
// The vote's type and the bit counts that match it. CUDA: 32-bit. HIP: every _sync intrinsic
// static_asserts a 64-bit mask and __ballot_sync returns 64 bits (infra-facts B3); __popc or
// __ffs on such a mask would silently see only its low 32 lanes (ch12-facts B1).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
static __device__ unsigned int popcMask(lane_mask_t m) { return __popcll(m); }
static __device__ unsigned int firstLane(lane_mask_t m) { return __ffsll((long long)m) - 1; }   // m != 0
#else
typedef unsigned int lane_mask_t;
static __device__ unsigned int popcMask(lane_mask_t m) { return __popc(m); }
static __device__ unsigned int firstLane(lane_mask_t m) { return __ffs((int)m) - 1; }           // m != 0
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
static __device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
static __device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
static __device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// Lab 14.3's key, ascending: a larger logit has a larger key as an unsigned integer. -0.0
// becomes +0.0 first; then a non-negative float gets its sign bit set and a negative one has
// every bit inverted (CUB's documented transform, ch14-facts A3).
static __device__ unsigned int orderedKey(unsigned int u) {
    if (u == 0x80000000u) {
        u = 0u;
    }
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// ---------------------------------------------------------------- 1. the threshold
__global__ void topkThreshold(const unsigned int* logitBits, unsigned int* scratch, int cols, int k) {
    __shared__ unsigned int hist_s[BINS];
    __shared__ unsigned int pick_s[2];   // the crossing bin, and the keys in the bins above it

    const lane_mask_t full = ~(lane_mask_t)0;
    const unsigned int n = (unsigned int)cols;
    const unsigned int* row = logitBits + (size_t)blockIdx.x * n;   // block b owns row b

    // The same three values in every thread: they are computed from shared memory only.
    unsigned int prefix = 0;   // the decided bits of K*
    unsigned int mask = 0;     // which bits are decided
    unsigned int need = (unsigned int)k < n ? (unsigned int)k : n;   // keys still to place

    for (int shift = 32 - DIGIT_BITS; shift >= 0; shift -= DIGIT_BITS) {
        // (a) Count the current byte of every key still undecided: chapter 9's privatised
        //     histogram, one per block, in shared memory.
        for (unsigned int b = threadIdx.x; b < BINS; b += blockDim.x) {
            hist_s[b] = 0u;
        }
        __syncthreads();
        for (unsigned int c = threadIdx.x; c < n; c += blockDim.x) {   // coalesced reads
            const unsigned int key = orderedKey(row[c]);
            if ((key & mask) == prefix) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bin(hist_s[(key >> shift) & (BINS - 1u)]);
                bin.fetch_add(1u, cuda::memory_order_relaxed);
            }
        }
        __syncthreads();

        // (b) Walk the bins from the top, one warp, warpSize bins at a time: an inclusive warp
        //     scan of the counts (Fig. 11.8) plus the keys of the bins already walked. The
        //     first lane whose running count reaches `need` holds the crossing bin.
        if (warpIdx() == 0) {
            unsigned int above = 0;
            for (unsigned int base = 0; base < BINS; base += warpSize) {
                const unsigned int b = BINS - 1u - (base + laneIdx());
                const unsigned int count = hist_s[b];
                unsigned int incl = count;
                for (unsigned int s = 1; s < (unsigned int)warpSize; s *= 2) {
                    const unsigned int left = __shfl_up_sync(full, incl, s);
                    if (laneIdx() >= s) {
                        incl += left;
                    }
                }
                const lane_mask_t reached = __ballot_sync(full, above + incl >= need);
                if (reached != 0) {   // the same in every lane: the warp leaves together
                    if (laneIdx() == firstLane(reached)) {
                        pick_s[0] = b;
                        pick_s[1] = above + incl - count;
                    }
                    break;
                }
                above += __shfl_sync(full, incl, warpSize - 1);
            }
        }
        __syncthreads();

        // (c) Extend the prefix by the crossing bin's byte; the keys above it are placed.
        const unsigned int bin = pick_s[0];
        const unsigned int inBin = hist_s[bin];
        need -= pick_s[1];
        prefix |= bin << shift;
        mask |= (BINS - 1u) << shift;
        __syncthreads();   // every thread has read hist_s and pick_s before the next pass
        if (need == inBin) {
            break;         // every key of the crossing bin is needed: the set is decided
        }
    }

    if (threadIdx.x == 0) {
        unsigned int* out = scratch + (size_t)blockIdx.x * SCRATCH_PER_ROW;
        out[0] = prefix;   // K*
        out[1] = need;     // ties at K* still needed
        out[2] = 0u;
        out[3] = 0u;
    }
}

// ---------------------------------------------------------------- 2. the gather
__global__ void topkGather(const unsigned int* logitBits, const unsigned int* scratch, int* outIdx,
                           unsigned int* outBits, int cols, int k) {
    __shared__ unsigned int warpTies_s[MAX_WARPS];    // ties per warp, this round
    __shared__ unsigned int warpKeeps_s[MAX_WARPS];   // kept keys per warp, this round

    const lane_mask_t full = ~(lane_mask_t)0;
    const lane_mask_t lanesBelow = ((lane_mask_t)1 << laneIdx()) - 1;   // bits 0 .. lane-1
    const unsigned int n = (unsigned int)cols;
    const unsigned int* row = logitBits + (size_t)blockIdx.x * n;
    int* idxRow = outIdx + (size_t)blockIdx.x * (size_t)k;        // the output's row stride is k
    unsigned int* valRow = outBits + (size_t)blockIdx.x * (size_t)k;
    const unsigned int* thr = scratch + (size_t)blockIdx.x * SCRATCH_PER_ROW;
    const unsigned int kStar = thr[0], need = thr[1];
    const unsigned int total = (unsigned int)k < n ? (unsigned int)k : n;

    // Running counts over the rounds already done; the same in every thread.
    unsigned int tiesBefore = 0, keptBefore = 0;

    for (unsigned int base = 0; base < n && keptBefore < total; base += blockDim.x) {
        const unsigned int c = base + threadIdx.x;
        unsigned int u = 0;
        bool above = false, tie = false;
        if (c < n) {
            u = row[c];
            const unsigned int key = orderedKey(u);
            above = key > kStar;
            tie = key == kStar;
        }

        // Rank of each tie among the row's ties, in column order.
        const lane_mask_t tieVote = __ballot_sync(full, tie);
        if (laneIdx() == 0) {
            warpTies_s[warpIdx()] = popcMask(tieVote);
        }
        __syncthreads();
        unsigned int tieOffset = 0, roundTies = 0;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            const unsigned int t = warpTies_s[w];
            tieOffset += w < warpIdx() ? t : 0u;
            roundTies += t;
        }
        const bool keep = above || (tie && tiesBefore + tieOffset + popcMask(tieVote & lanesBelow) < need);

        // Slot of each kept key: the kept keys before it, in column order.
        const lane_mask_t keepVote = __ballot_sync(full, keep);
        if (laneIdx() == 0) {
            warpKeeps_s[warpIdx()] = popcMask(keepVote);
        }
        __syncthreads();   // also: every read of warpTies_s is done before the next round's write
        unsigned int keepOffset = 0, roundKept = 0;
        for (unsigned int w = 0; w < warpsInBlock(); ++w) {
            const unsigned int t = warpKeeps_s[w];
            keepOffset += w < warpIdx() ? t : 0u;
            roundKept += t;
        }
        if (keep) {
            const unsigned int slot = keptBefore + keepOffset + popcMask(keepVote & lanesBelow);
            idxRow[slot] = (int)c;
            valRow[slot] = u;
        }
        // The next round writes warpKeeps_s only after its first barrier, which every thread
        // reaches after reading warpKeeps_s here.
        tiesBefore += roundTies;
        keptBefore += roundKept;
    }
}

// ---------------------------------------------------------------- the launcher
void launchTopKRows(const float* logits, int* outIdx, float* outVal, unsigned int* scratch, int rows, int cols,
                    int k) {
    if (rows <= 0 || cols <= 0 || k <= 0) return;
    // The kernels see the float32 tensors as 32-bit unsigned integers (README: why).
    const unsigned int* logitBits = (const unsigned int*)logits;
    topkThreshold<<<rows, BLOCK_SIZE>>>(logitBits, scratch, cols, k);
    topkGather<<<rows, BLOCK_SIZE>>>(logitBits, scratch, outIdx, (unsigned int*)outVal, cols, k);
}
