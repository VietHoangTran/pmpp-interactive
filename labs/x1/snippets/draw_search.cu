// The final draw of milestone M5, one block per row (section X1.8). Ours: sampling is not
// in the book. It is built from the two-stage reduction of section 10.8, the warp-level
// block scan of section 11.4 and the warp vote of section 12.3.
//
// Row r holds counts[r] >= 1 survivor weights w_i >= 0 at weights[r * stride]; u[r] is in
// [0, 1). The kernel writes the position of the first survivor whose inclusive sum is
// strictly greater than u * total, or, when rounding leaves no sum above that target, the
// last survivor with a positive weight. The caller maps the position to a token id.
//
// What makes it portable (notes/today/x1-facts.md and the facts files it cites):
//   - a warp's width is read from warpSize, never assumed (32 on NVIDIA, 64 or 32 on AMD);
//   - the vote's type follows the backend: HIP's ballots are 64 bits, and __ffs would see
//     only the low 32 lanes of one (ch12-facts B1), so HIP uses __ffsll;
//   - one block per row: no block ever waits for another (Metal documents no forward
//     progress between threadgroups, ch11-facts C5).
// The barrier comments name the two barriers section X1.8 asks you to remove.

#ifndef DRAW_BLOCK
#define DRAW_BLOCK 256                         // threads per block; a multiple of the warp width
#endif
#define VEC 4                                  // survivors per thread per chunk
#define MIN_WARP_SIZE 32                       // the narrowest warp of any backend here
#define MAX_WARPS (DRAW_BLOCK / MIN_WARP_SIZE)

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
__device__ int firstLane(lane_mask_t m) { return __ffsll((long long)m) - 1; }
#else
typedef unsigned int lane_mask_t;
__device__ int firstLane(lane_mask_t m) { return __ffs((int)m) - 1; }
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// Lane 0 ends with the warp's sum (the tree of Fig. 10.13, read through warpSize).
__device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int offset = warpSize / 2; offset > 0; offset /= 2)
        v += __shfl_down_sync(full, v, offset);
    return v;
}

// Inclusive Kogge-Stone scan across the warp (Fig. 11.8, read through warpSize).
__device__ float warpInclusiveScan(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (int stride = 1; stride < warpSize; stride *= 2) {
        const float left = __shfl_up_sync(full, v, stride);
        if (laneIdx() >= (unsigned int)stride) v += left;
    }
    return v;
}

__global__ void drawKernel(const float* weights, const int* counts, const float* u, int* tokens, int stride) {
    __shared__ float scratch_s[MAX_WARPS];   // warp sums, then the total, then warp totals
    __shared__ int winner_s;
    const lane_mask_t full = ~(lane_mask_t)0;
    const float* w = weights + (size_t)blockIdx.x * stride;
    const int n = counts[blockIdx.x];

    // 1. The row's total: a strided sum per thread, a sum per warp, then warp 0 (section 10.8).
    float part = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) part += w[i];
    part = warpSum(part);
    if (laneIdx() == 0) scratch_s[warpIdx()] = part;
    __syncthreads();
    if (warpIdx() == 0) {
        float v = laneIdx() < warpsInBlock() ? scratch_s[laneIdx()] : 0.0f;
        v = warpSum(v);
        if (laneIdx() == 0) {
            scratch_s[0] = v;
            winner_s = n;                    // "no survivor found yet"
        }
    }
    __syncthreads();                         // barrier A: the total is written
    const float target = u[blockIdx.x] * scratch_s[0];
    __syncthreads();                         // barrier B: every thread has read it

    // 2. Walk the row in chunks of blockDim.x * VEC survivors, VEC consecutive ones per thread.
    float aggregate = 0.0f;                  // the sum of every chunk before this one
    for (int base = 0; base < n; base += blockDim.x * VEC) {
        const int first = base + threadIdx.x * VEC;
        float local[VEC];
        float run = 0.0f;
        for (int j = 0; j < VEC; ++j) {
            run += (first + j < n) ? w[first + j] : 0.0f;
            local[j] = run;                  // inclusive over this thread's survivors
        }
        const float incl = warpInclusiveScan(run);
        float excl = __shfl_up_sync(full, incl, 1);
        if (laneIdx() == 0) excl = 0.0f;
        if (laneIdx() == warpSize - 1) scratch_s[warpIdx()] = incl;   // the warp's total
        __syncthreads();                     // every warp's total is in scratch_s

        float carry = 0.0f, chunk = 0.0f;    // warps before mine; the whole chunk
        for (unsigned int k = 0; k < warpsInBlock(); ++k) {
            const float s = scratch_s[k];
            if (k < warpIdx()) carry += s;
            chunk += s;
        }
        if (aggregate + chunk > target) {    // the chunk crosses the target: find where
            const float before = aggregate + (carry + excl);
            int hit = -1;                    // my first survivor past the target
            for (int j = VEC - 1; j >= 0; --j)
                if (first + j < n && before + local[j] > target) hit = j;
            const lane_mask_t vote = __ballot_sync(full, hit >= 0);
            if (hit >= 0 && (int)laneIdx() == firstLane(vote))
                atomicMin(&winner_s, first + hit);   // the lowest flagged warp wins
        }
        __syncthreads();                     // the vote is in; scratch_s may be reused
        if (winner_s < n) break;             // the same decision in every thread
        aggregate += chunk;
    }

    // 3. Rounding can leave no sum above the target: take the last positive weight.
    if (threadIdx.x == 0) {
        int k = winner_s;
        if (k >= n) {
            k = n - 1;
            while (k > 0 && !(w[k] > 0.0f)) --k;
        }
        tokens[blockIdx.x] = k;
    }
}

void launchDraw(const float* weights, const int* counts, const float* u, int* tokens, int rows, int stride) {
    drawKernel<<<rows, DRAW_BLOCK>>>(weights, counts, u, tokens, stride);
}
