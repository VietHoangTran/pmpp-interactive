// PROVIDED, do not edit. Min-p on a sorted row, and how many candidates survive both top-p and
// min-p (used by lab X1.3; ours: sampling is not in the book).
//
// On a row sorted in descending order, min-p keeps a prefix: the logits at least the first one
// plus ln(minP) (section X1.6; llama.cpp computes `max_logit + logf(p)` the same way, in float,
// with ln(minP) on the host, notes/today/ch12-facts.md D3). Lab 11.6 already counted top-p's
// prefix in kept[r]. Both are prefixes of the same row, so the survivors are the shorter one,
// and never fewer than one token:
//     n[r] = max(1, min(kept[r], countMinP[r]))
// minP <= 0 disables min-p (countMinP = cols), as llama.cpp's 0.0 does (x1-facts D1).
//
// The count is the two-stage reduction of chapter 10 (Fig. 10.16, p. 243) on integers. Every
// logit passing the test is counted, not only a leading run: on a sorted row the two agree.
#include <cmath>

#define MINP_BLOCK 256                            // threads per block; one block per row
#define MINP_MIN_WARP 32                          // the narrowest warp any backend here has
#define MINP_MAX_WARPS (MINP_BLOCK / MINP_MIN_WARP)

#if defined(__HIPCC__)
typedef unsigned long long minp_mask_t;           // HIP requires 64-bit masks (infra-facts B3)
#else
typedef unsigned int minp_mask_t;
#endif

static __device__ int warpSumMinP(int v) {
    const minp_mask_t full = ~(minp_mask_t)0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(full, v, offset);
    }
    return v;
}

__global__ void minPCombineKernel(const float* sortedLogits, const int* kept, int* n, int cols, float lnMinP,
                                  int useMinP) {
    __shared__ int warpCounts_s[MINP_MAX_WARPS];
    const float* row = sortedLogits + (size_t)blockIdx.x * cols;
    const unsigned int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
    const unsigned int warps = (blockDim.x + warpSize - 1) / warpSize;

    int count = 0;
    if (useMinP) {
        const float threshold = row[0] + lnMinP;   // one float addition, as llama.cpp's host does
        for (int i = threadIdx.x; i < cols; i += blockDim.x) {
            count += row[i] >= threshold ? 1 : 0;
        }
    }
    count = warpSumMinP(count);
    if (lane == 0) {
        warpCounts_s[warp] = count;
    }
    __syncthreads();
    if (warp == 0) {
        int total = lane < warps ? warpCounts_s[lane] : 0;
        total = warpSumMinP(total);
        if (lane == 0) {
            const int countMinP = useMinP ? total : cols;
            const int k = kept[blockIdx.x] < countMinP ? kept[blockIdx.x] : countMinP;
            n[blockIdx.x] = k > 1 ? k : 1;
        }
    }
}

// The host computes ln(minP) once, in float, as llama.cpp does.
void launchMinPCombine(const float* sortedLogits, const int* kept, int* n, int rows, int cols, float minP) {
    if (rows <= 0 || cols <= 0) return;
    const int useMinP = minP > 0.0f ? 1 : 0;
    const float lnMinP = useMinP ? logf(minP) : 0.0f;
    minPCombineKernel<<<rows, MINP_BLOCK>>>(sortedLogits, kept, n, cols, lnMinP, useMinP);
}
