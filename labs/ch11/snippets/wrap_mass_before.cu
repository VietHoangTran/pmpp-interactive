// A kernel of our own for chapter 11's summary page (the mixed check). For each row of
// sorted probabilities, one block per row and at most blockDim.x tokens per row, it writes
// the mass ranked strictly before each token: the exclusive scan that top-p compares with p
// (lab 11.6). It turns lab 11.1's inclusive block scan into an exclusive one by taking the
// inclusive value one lane to the left. It has one defect on purpose; the page asks the
// reader to find it.
// The defines, the mask type, laneIdx, warpIdx, warpScan and blockScan are lab 11.1's
// (labs/ch11/lab1_blockscan/solution/kernel.cu). app/src/chapters/ch11/Wrap.snippets.test.ts
// fails if they drift apart, and runs this kernel on the course's emulator through
// check/wrap_mass_before_test.cu.
#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (ch11-facts B1)

// The mask of __shfl_up_sync. CUDA: unsigned. HIP: every _sync intrinsic static_asserts
// a 64-bit mask, in wave32 mode too (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

// Every lane of a warp calls this with its value and gets the inclusive scan of the
// warp's values back. A lane below `stride` gets its own value from __shfl_up_sync
// (ch11-facts A3), so the lane test is what keeps it from adding itself.
__device__ __forceinline__ float warpScan(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;   // every lane of the warp
    for (unsigned int stride = 1; stride < (unsigned int)warpSize; stride *= 2) {
        float leftVal = __shfl_up_sync(full, val, stride);
        if (laneIdx() >= stride) {
            val += leftVal;
        }
    }
    return val;
}

// Scan-scan-add over the warps of the block. Every thread of the block must call it: it
// holds two barriers and full-mask shuffles. warpSums_s has room for one float per warp.
__device__ __forceinline__ float blockScan(float val, float* warpSums_s) {
    // Stage 1: every warp scans its own values.
    val = warpScan(val);
    if (laneIdx() == (unsigned int)warpSize - 1) {
        warpSums_s[warpIdx()] = val;                  // the warp's total
    }
    __syncthreads();

    // Stage 2: warp 0 scans the warp totals. Lanes with no warp behind them feed 0.
    if (warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float warpSum = laneIdx() < warps ? warpSums_s[laneIdx()] : 0.0f;
        warpSum = warpScan(warpSum);
        if (laneIdx() < warps) {
            warpSums_s[laneIdx()] = warpSum;
        }
    }
    __syncthreads();

    // Stage 3: every warp but the first adds the total of the warps before it.
    if (warpIdx() > 0) {
        val += warpSums_s[warpIdx() - 1];
    }
    return val;
}

__global__ void massBeforeRows(const float* probs, float* massBefore, unsigned int cols) {
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    const float* row = probs + blockIdx.x * cols;
    float* out = massBefore + blockIdx.x * cols;

    float p = (threadIdx.x < cols) ? row[threadIdx.x] : 0.0f;
    float inclusive = blockScan(p, warpSums_s);              // every thread, in range or not

    const lane_mask_t full = ~(lane_mask_t)0;
    float before = __shfl_up_sync(full, inclusive, 1);       // the inclusive mass one lane to the left
    if (threadIdx.x == 0) {
        before = 0.0f;                                       // nothing is ranked before token 0
    }
    if (threadIdx.x < cols) {
        out[threadIdx.x] = before;
    }
}
