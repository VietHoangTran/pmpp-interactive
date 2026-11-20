// Lab X1.4 (CUDA; also built for HIP and for the CPU emulators): lab X1.2's fused top-p and
// min-p kernel, filterSortedFused, with three independent defects. It compiles. Find and fix
// all three; README.md has the method. launchFilterSortedFused at the bottom is correct.
//
// Not in the book: module X1 is ours (PLAN.md). The pieces are the book's: chapter 10's
// two-stage reduction (Figs. 10.13 and 10.16, pp. 240 and 243) and chapter 11's warp and block
// scans (Figs. 11.8 and 11.9, p. 264) with the coarsened row walk of lab 11.3. The order of the
// filters, the top-p rule and the min-p test are llama.cpp's (notes/today/x1-facts.md D1,
// ch11-facts D2, ch12-facts D3); the fusion is ours.
//
// Contract (lab X1.2's part B; README.md). Every row holds K finite logits sorted in descending
// order; one block of BLOCK_DIM threads per row. For each row:
//   sortedProbs = softmax at temperature 1 of the row;
//   n = max(1, min(a, b)) where
//     a = the number of tokens i with i == 0, or topP >= 1, or (mass ranked before i) < topP
//     b = the number of tokens with logit >= logit[0] + ln(minP), ln(minP) computed once on the
//         host as a float; every token when minP <= 0.

#include <cfloat>
#include <cmath>

// The course's CUDA front-end check (labs/pmpp.py) declares no CUDA Math API functions, so it
// is given expf's device declaration here, as in lab 10.3. nvcc, hipcc and the emulator never
// see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
#endif

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

// ---------------------------------------------------------------- building blocks
// Static, as in lab X1.2, where provided copies define functions with some of these names.

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

        // (b) This lane's run of COARSE_FACTOR consecutive tokens: running sums in registers.
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            threadSum += myStage[lane * COARSE_FACTOR + c];
            run[c] = threadSum;
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
        const float before = (warp > 0 ? warpSums_s[warp - 1] : 0.0f) + exclusive;
        const unsigned int first = chunk + warpSlice + lane * COARSE_FACTOR;
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            const unsigned int j = first + c;
            if (j < K) {
                float mass = before + run[c];
                keptTopP += (j == 0 || keepAll || mass < topP) ? 1 : 0;
                mass += carry;
            }
        }
        carry += warpSums_s[numWarps - 1];
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
    // Once per call, on the host, in float: llama.cpp's logf(p) (ch12-facts D3).
    const float lnMinP = minP > 0.0f ? logf(minP) : 0.0f;
    filterSortedFused<<<rows, BLOCK_DIM>>>(sortedLogits, sortedProbs, n, (unsigned int)K, topP, minP, lnMinP);
}
