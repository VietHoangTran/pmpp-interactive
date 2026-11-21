// Lab X1.2 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5: the top-p and min-p stage of the sampler, first composed from earlier labs,
// then fused into one kernel.
//
// Not in the book: module X1 is ours (PLAN.md). The pieces are the book's: chapter 10's
// two-stage reduction (Figs. 10.13 and 10.16, pp. 240 and 243) and chapter 11's warp and block
// scans (Figs. 11.8 and 11.9, p. 264) with the coarsened row walk of lab 11.3. The order of the
// two filters (top-p, then min-p) is llama.cpp's default chain (notes/today/x1-facts.md D1),
// its min-p test on logits is llama.cpp's host sampler (ch12-facts D3), and its top-p rule keeps
// the token that crosses p (ch11-facts D2). The fusion and the prefix argument below are ours.
//
// Contract (README.md states it in full). Every row holds K finite logits; topP and minP are
// floats; rows >= 1, K >= 1; one block of BLOCK_DIM threads per row.
//   Part A, launchFilterRows: sort each row of candLogits in descending order, stably (ties by
//     lower position), into sortedPos (the positions) and sortedLogits (the same 32 bits);
//     sortedProbs = softmax at temperature 1 of the sorted row; n[r] = max(1, min(a, b)) where
//       a = the number of tokens i with i == 0, or topP >= 1, or (mass ranked before i) < topP
//       b = the number of tokens with sortedLogits[i] >= sortedLogits[0] + ln(minP), with
//           ln(minP) computed once on the host as a float; every token when minP <= 0.
//   Part B, launchFilterSortedFused: the same sortedProbs and n from rows already sorted.
// Both filters keep a prefix of the sorted row, so the tokens that survive both are the
// shorter prefix: n of them. That is why n is a minimum and not a second pass (section X1.6).

#include <cfloat>
#include <cmath>

#define BLOCK_DIM 256                           // threads per block; one block per row
#define COARSE_FACTOR 4                         // tokens per thread per chunk (lab 11.6)
#define CHUNK (COARSE_FACTOR * BLOCK_DIM)       // tokens per chunk: 1,024
#define MIN_WARP_SIZE 32                        // the narrowest warp a CUDA or HIP device has
#define MAX_WARPS (BLOCK_DIM / MIN_WARP_SIZE)   // enough per-warp slots on every backend

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;         // HIP requires 64-bit masks (infra-facts B3)
#else
typedef unsigned int lane_mask_t;
#endif

// ---------------------------------------------------------------- provided by earlier labs
// Their reference solutions, compiled from copies in ../shared-g12/ (lab 14.3's sort, lab
// 10.3's softmax, lab 11.6's top-p mask). Contracts: their README files.
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch,
                        int rows, int cols);
void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale);
void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                    float p);

// ---------------------------------------------------------------- building blocks
// Every helper is static: the provided copies define functions with some of these names, and
// the emulator links all four .cu files into one program.

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
static __device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
static __device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

// Fig. 10.13 on floats and on ints: lane 0 returns the warp's sum.
static __device__ float warpSum(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

static __device__ int warpSumInt(int val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

// Fig. 11.8, fixed in lab 11.3: inclusive scan across the lanes of one warp.
static __device__ float warpScan(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = 1; stride < warpSize; stride *= 2) {
        float left = __shfl_up_sync(full, val, stride);
        if (laneIdx() >= stride) {
            val += left;
        }
    }
    return val;
}

// Fig. 10.16, as lab 10.3 extends it: the block's sum, returned to every thread through
// scratch[0]. scratch has MAX_WARPS entries. Every thread must call it.
static __device__ float blockSum(float val, float* scratch) {
    const unsigned int numWarps = blockDim.x / warpSize;
    val = warpSum(val);
    if (laneIdx() == 0) {
        scratch[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        val = laneIdx() < numWarps ? scratch[laneIdx()] : 0.0f;
        val = warpSum(val);
        if (laneIdx() == 0) {
            scratch[0] = val;
        }
    }
    __syncthreads();
    return scratch[0];
}

// ---------------------------------------------------------------- part A: min-p, then n
// One block per row of sorted logits. Counts the tokens min-p keeps and combines that count
// with the top-p count lab 11.6 wrote: n[row] = max(1, min(keptTopP[row], count)).
__global__ void minPCount(const float* sortedLogits, const int* keptTopP, int* n, int K, float minP,
                          float lnMinP) {
    __shared__ int warpCounts_s[MAX_WARPS];
    const float* row = sortedLogits + (size_t)blockIdx.x * K;

    // The row is sorted, so its largest logit is element 0. llama.cpp keeps a token when its
    // logit is at least max + logf(p) (ch12-facts D3). minP <= 0 keeps every token: -FLT_MAX is
    // below every finite logit, and unlike -INFINITY it is defined under Metal's fast math
    // (ch10-facts C3), so the three backends share one rule.
    const float threshold = minP > 0.0f ? row[0] + lnMinP : -FLT_MAX;
    int count = 0;
    for (int c = threadIdx.x; c < K; c += blockDim.x) {
        count += row[c] >= threshold ? 1 : 0;
    }

    // Chapter 10's two-stage reduction on the threads' counts.
    const unsigned int numWarps = blockDim.x / warpSize;
    count = warpSumInt(count);
    if (laneIdx() == 0) {
        warpCounts_s[warpIdx()] = count;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        int total = laneIdx() < numWarps ? warpCounts_s[laneIdx()] : 0;
        total = warpSumInt(total);
        if (laneIdx() == 0) {
            const int a = keptTopP[blockIdx.x];
            const int both = a < total ? a : total;   // both filters keep a prefix: the shorter one
            n[blockIdx.x] = both > 1 ? both : 1;      // the largest logit always survives
        }
    }
}

void launchMinPCount(const float* sortedLogits, const int* keptTopP, int* n, int rows, int K, float minP) {
    if (rows <= 0 || K <= 0) return;
    // Once per call, on the host, in float: llama.cpp's logf(p) (ch12-facts D3). Every backend
    // then adds the same float to the row's maximum.
    const float lnMinP = minP > 0.0f ? logf(minP) : 0.0f;
    minPCount<<<rows, BLOCK_DIM>>>(sortedLogits, keptTopP, n, K, minP, lnMinP);
}

// ---------------------------------------------------------------- part A: the composition
// workspace, in this order (README.md):
//   sort scratch  4 * rows * K unsigned ints   (lab 14.3: keys and indices, twice)
//   massBefore    rows * K floats              (lab 11.6)
//   keptTopP      rows ints                    (lab 11.6)
//   keep          rows * K bytes               (lab 11.6), last, so nothing after it needs alignment
void launchFilterRows(const float* candLogits, int* sortedPos, float* sortedLogits, float* sortedProbs, int* n,
                      void* workspace, int rows, int K, float topP, float minP) {
    if (rows <= 0 || K <= 0) return;
    const size_t cells = (size_t)rows * (size_t)K;
    unsigned int* sortScratch = (unsigned int*)workspace;
    float* massBefore = (float*)(sortScratch + 4 * cells);
    int* keptTopP = (int*)(massBefore + cells);
    unsigned char* keep = (unsigned char*)(keptTopP + rows);

    // 1. Lab 14.3: descending and stable, so equal logits stay in position order. Its key
    //    transform orders any float; these are logits instead of probabilities.
    launchSortRowsDesc(candLogits, sortedLogits, sortedPos, sortScratch, rows, K);
    // 2. Lab 10.3 at scale 1. Softmax is monotonic, so the probabilities are sorted too.
    launchSoftmaxRows(sortedLogits, sortedProbs, rows, K, 1.0f);
    // 3. Lab 11.6: the mass before each token, the keep mask and its count.
    launchTopPMask(sortedProbs, massBefore, keep, keptTopP, rows, K, topP);
    // 4. Min-p on the sorted logits, and n.
    launchMinPCount(sortedLogits, keptTopP, n, rows, K, minP);
}

// ---------------------------------------------------------------- part B: fused
// One block per row of sorted logits. Reads the row twice and writes sortedProbs once.
//   pass 1: the softmax denominator. The maximum is element 0: the row is sorted.
//   pass 2: lab 11.6's chunked scan of the probabilities with a carried prefix. Each token's
//           probability is computed, written and staged; its min-p decision is made on its logit
//           while that is in a register, and its top-p decision on its mass while that is.
__global__ void filterSortedFused(const float* sortedLogits, float* sortedProbs, int* n, unsigned int K,
                                  float topP, float minP, float lnMinP) {
    __shared__ float stage_s[CHUNK];                 // each warp's slice of the chunk
    __shared__ float warpSums_s[MAX_WARPS];          // the block's sum, then each chunk's warp totals
    __shared__ int warpTopP_s[MAX_WARPS];
    __shared__ int warpMinP_s[MAX_WARPS];

    const size_t rowStart = (size_t)blockIdx.x * K;
    const float* rowLogits = sortedLogits + rowStart;
    float* rowProbs = sortedProbs + rowStart;

    const lane_mask_t full = ~(lane_mask_t)0;
    const unsigned int lane = laneIdx();
    const unsigned int warp = warpIdx();
    const unsigned int numWarps = BLOCK_DIM / warpSize;
    const unsigned int warpSlice = warp * COARSE_FACTOR * warpSize;
    float* myStage = stage_s + warpSlice;

    const float l0 = rowLogits[0];                   // the row's maximum
    const float threshold = minP > 0.0f ? l0 + lnMinP : -FLT_MAX;
    const bool keepAll = topP >= 1.0f;               // llama.cpp's early return (ch11-facts D2)

    // Pass 1: sum of exp(l - l0), as lab 10.3 computes it (every term in (0, 1], the first is 1).
    float s = 0.0f;
    for (unsigned int c = threadIdx.x; c < K; c += BLOCK_DIM) {
        s += expf(rowLogits[c] - l0);
    }
    const float total = blockSum(s, warpSums_s);
    // Every thread has just read the total from warpSums_s[0], and the first chunk's scan writes
    // its warp totals into the same array. Without this barrier a fast warp can overwrite the
    // total before a slow warp has read it (lab 10.3 has the same barrier for the same reason).
    __syncthreads();
    const float inv = 1.0f / total;

    int keptTopP = 0;     // tokens of this thread that top-p keeps
    int keptMinP = 0;     // tokens of this thread that min-p keeps
    float carry = 0.0f;   // the mass of every earlier chunk of this row

    for (unsigned int chunk = 0; chunk < K; chunk += CHUNK) {
        // (a) Coalesced: each warp reads its slice of logits, writes the probabilities, and
        //     stages them for the scan. Min-p is decided here, on the logit.
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            const unsigned int j = chunk + warpSlice + c * warpSize + lane;
            float prob = 0.0f;
            if (j < K) {
                const float l = rowLogits[j];
                prob = expf(l - l0) * inv;
                rowProbs[j] = prob;
                keptMinP += l >= threshold ? 1 : 0;
            }
            myStage[c * warpSize + lane] = prob;
        }
        __syncwarp();

        // (b) This lane's run of COARSE_FACTOR consecutive tokens: exclusive sums in registers.
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            run[c] = threadSum;
            threadSum += myStage[lane * COARSE_FACTOR + c];
        }

        // (c) Block scan of the run totals: warp scans, then warp 0 scans the warp totals.
        const float inclusive = warpScan(threadSum);
        float exclusive = __shfl_up_sync(full, inclusive, 1);
        if (lane == 0) {
            exclusive = 0.0f;
        }
        if (lane == warpSize - 1) {
            warpSums_s[warp] = inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            float warpTotal = lane < numWarps ? warpSums_s[lane] : 0.0f;
            warpTotal = warpScan(warpTotal);
            if (lane < numWarps) {
                warpSums_s[lane] = warpTotal;
            }
        }
        __syncthreads();

        // (d) The mass ranked before each token of the run, earlier chunks included, and the
        //     top-p decision on it (lab 11.6's rule: the token that crosses p is kept).
        const float before = carry + (warp > 0 ? warpSums_s[warp - 1] : 0.0f) + exclusive;
        carry += warpSums_s[numWarps - 1];
        const unsigned int first = chunk + warpSlice + lane * COARSE_FACTOR;
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            const unsigned int j = first + c;
            if (j < K) {
                const float mass = before + run[c];
                keptTopP += (j == 0 || keepAll || mass < topP) ? 1 : 0;
            }
        }
        __syncthreads();   // every read of warpSums_s and stage_s is done before the next chunk writes them
    }

    // n = max(1, min(top-p count, min-p count)): both counts in one two-stage reduction.
    keptTopP = warpSumInt(keptTopP);
    keptMinP = warpSumInt(keptMinP);
    if (lane == 0) {
        warpTopP_s[warp] = keptTopP;
        warpMinP_s[warp] = keptMinP;
    }
    __syncthreads();
    if (warp == 0) {
        int a = lane < numWarps ? warpTopP_s[lane] : 0;
        int b = lane < numWarps ? warpMinP_s[lane] : 0;
        a = warpSumInt(a);
        b = warpSumInt(b);
        if (lane == 0) {
            const int both = a < b ? a : b;
            n[blockIdx.x] = both > 1 ? both : 1;
        }
    }
}

void launchFilterSortedFused(const float* sortedLogits, float* sortedProbs, int* n, int rows, int K, float topP,
                             float minP) {
    if (rows <= 0 || K <= 0) return;
    const float lnMinP = minP > 0.0f ? logf(minP) : 0.0f;   // as in launchMinPCount
    filterSortedFused<<<rows, BLOCK_DIM>>>(sortedLogits, sortedProbs, n, (unsigned int)K, topP, minP, lnMinP);
}
