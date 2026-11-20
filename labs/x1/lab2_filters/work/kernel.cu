// Lab X1.2 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M5: the top-p and min-p stage of the sampler, first composed from earlier labs,
// then fused into one kernel. README.md has the task, the contract and the tests.
//
// Not in the book: module X1 is ours. The pieces are the book's: chapter 10's two-stage
// reduction (Figs. 10.13 and 10.16, pp. 240 and 243) and chapter 11's warp and block scans
// (Figs. 11.8 and 11.9, p. 264) with lab 11.3's coarsened row walk.
//
// Contract, in short (README.md states it in full). Every row holds K finite logits.
//   Part A, launchFilterRows: sort each row of candLogits in descending order, stably, into
//     sortedPos and sortedLogits; sortedProbs = softmax at temperature 1 of the sorted row;
//     n[r] = max(1, min(a, b)) where
//       a = the number of tokens i with i == 0, or topP >= 1, or (mass ranked before i) < topP
//       b = the number of tokens with sortedLogits[i] >= sortedLogits[0] + ln(minP), with
//           ln(minP) computed once on the host as a float; every token when minP <= 0.
//   Part B, launchFilterSortedFused: the same sortedProbs and n from rows already sorted.

#include <cfloat>
#include <cmath>

// The course's CUDA front-end check declares no CUDA Math API functions, so it is given
// expf's device declaration here. nvcc, hipcc and the emulator never see it.
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

// ---------------------------------------------------------------- provided by earlier labs
// Their reference solutions, compiled from copies in ../shared-g12/ (lab 14.3's sort, lab
// 10.3's softmax, lab 11.6's top-p mask). Contracts: their README files.
void launchSortRowsDesc(const float* probs, float* sortedProbs, int* sortedIdx, unsigned int* scratch,
                        int rows, int cols);
void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale);
void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                    float p);

// ---------------------------------------------------------------- provided building blocks
// Every helper is static: the provided copies define functions with some of these names, and
// the emulator links all four .cu files into one program. Keep your own helpers static too.

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
__global__ void minPCount(const float* sortedLogits, const int* keptTopP, int* n, int K, float minP,
                          float lnMinP) {
    // TODO 1: block blockIdx.x handles row blockIdx.x of sortedLogits, which is sorted.
    //         Count the row's logits that min-p keeps (the contract's b). Settle minP <= 0
    //         first: every token survives, so choose a threshold that no finite logit is
    //         below and that Metal's fast math also defines (README.md, "Portability").
    //         Add up the threads' counts with chapter 10's two stages, then let one thread
    //         write n[blockIdx.x] from that count and keptTopP[blockIdx.x], lab 11.6's count.
}

void launchMinPCount(const float* sortedLogits, const int* keptTopP, int* n, int rows, int K, float minP) {
    // TODO 2: ln(minP) once, here on the host, as a float (logf, as llama.cpp does), then one
    //         block of BLOCK_DIM threads per row.
}

// ---------------------------------------------------------------- part A: the composition
void launchFilterRows(const float* candLogits, int* sortedPos, float* sortedLogits, float* sortedProbs, int* n,
                      void* workspace, int rows, int K, float topP, float minP) {
    // TODO 3: carve workspace into the buffers the provided launchers need (README.md gives
    //         the layout the driver allocates), then: the sort, the softmax (what scale is
    //         temperature 1?), the top-p mask, and your min-p count. Each provided launcher's
    //         argument order is in its declaration above; check which buffer each one reads.
}

// ---------------------------------------------------------------- part B: fused
__global__ void filterSortedFused(const float* sortedLogits, float* sortedProbs, int* n, unsigned int K,
                                  float topP, float minP, float lnMinP) {
    // TODO 4: block blockIdx.x handles row blockIdx.x, already sorted. Read the logits twice
    //         and write sortedProbs once:
    //         - pass 1: the softmax denominator (where is the row's maximum?);
    //         - pass 2: lab 11.6's walk in chunks of CHUNK tokens with a carried prefix. Each
    //           warp loads its slice coalesced, computes and writes each probability, and stages
    //           it for the scan; decide min-p while the logit is in a register, and top-p while
    //           each token's mass is (which mass does the contract compare with p?);
    //         - then the two counts, combined into n[blockIdx.x] as part A combines them.
    //         For every shared array you reuse, ask who reads it last and who writes it next.
}

void launchFilterSortedFused(const float* sortedLogits, float* sortedProbs, int* n, int rows, int K, float topP,
                             float minP) {
    // TODO 5: as launchMinPCount: ln(minP) on the host, one block of BLOCK_DIM threads per row.
}
