// PROVIDED, do not edit. A copy of labs/ch11/lab6_topp/solution/kernel.cu (lab 11.6: the top-p keep mask),
// linked into lab X1.3 beside other copies. The only change: helper functions are static, so
// that no two linked files define the same name. app/src/chapters/x1/copies-g11.test.ts checks
// that nothing else differs.
// Lab 11.6 reference solution (CUDA; also built for HIP and the CPU emulators).
// Milestone M5, part 1: the top-p (nucleus) cumulative mass and keep mask.
//
// Book: the exclusive scan (section 11.1, p. 253) computed with lab 11.3's row scan: the
// coarsened, register-tiled scan of Figs. 11.12 and 11.13 (pp. 268 and 271) on the warp and
// block scans of Figs. 11.8 and 11.9 (p. 264), one block per row with a carried prefix. The
// top-p rule, the keep mask and the count are not in the book: they are ours, fused into the
// scan so that the probabilities are read once.
//
// Contract (README.md):
//   probs is [rows x cols] float32, row-major, batch first; each row sorted in descending order
//   and summing to 1 within rounding. For every row r and token i:
//     massBefore[r][i] = probs[r][0] + ... + probs[r][i-1]        (massBefore[r][0] = 0)
//     keep[r][i]       = 1 exactly when i == 0, or p >= 1, or massBefore[r][i] < p; else 0
//     kept[r]          = the number of ones in keep[r]
//   That is the rule llama.cpp, vLLM and transformers implement: the token that crosses p is
//   kept (ch11-facts D2, D4). Ties are broken by position, that is, by the sort.
//
// Where this file differs from the book's kernels: as in lab 11.3 (typed lane mask, warpSize,
// bounds checks, exclusive scan, per-warp staging, shared arrays declared in the kernel).

#define BLOCK_DIM 256                           // threads per block; one block per row
#define COARSE_FACTOR 4                         // elements per thread per chunk (p. 269)
#define MIN_WARP_SIZE 32                        // the narrowest warp a CUDA or HIP device has
#define MAX_WARPS (BLOCK_DIM / MIN_WARP_SIZE)   // enough per-warp slots on every backend

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;         // HIP requires 64-bit masks (infra-facts B3)
#else
typedef unsigned int lane_mask_t;
#endif

static __device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
static __device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// Fig. 11.8, fixed in lab 11.3: inclusive scan across the lanes of one warp.
static __device__ __inline__ float warpScan(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = 1; stride < warpSize; stride *= 2) {
        float leftVal = __shfl_up_sync(full, val, stride);
        if (laneIdx() >= stride) {
            val += leftVal;
        }
    }
    return val;
}

// Fig. 10.13's warp reduction (lab 10.1) on integers. Lane 0 returns the warp's sum.
static __device__ __inline__ int warpSumInt(int val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

__global__ void TopPMaskKernel(const float* probs, float* massBefore, unsigned char* keep, int* kept,
                               unsigned int cols, float p) {
    __shared__ float stage_s[COARSE_FACTOR * BLOCK_DIM];   // each warp's slice of the chunk
    __shared__ float warpSums_s[MAX_WARPS];
    __shared__ int warpCounts_s[MAX_WARPS];

    const size_t rowStart = (size_t)blockIdx.x * cols;
    const float* rowProbs = probs + rowStart;
    float* rowMass = massBefore + rowStart;
    unsigned char* rowKeep = keep + rowStart;

    const lane_mask_t full = ~(lane_mask_t)0;
    const unsigned int lane = laneIdx();
    const unsigned int warp = warpIdx();
    const unsigned int numWarps = BLOCK_DIM / warpSize;
    const unsigned int chunkSize = COARSE_FACTOR * BLOCK_DIM;
    const unsigned int warpSlice = warp * COARSE_FACTOR * warpSize;
    float* myStage = stage_s + warpSlice;

    // p >= 1 keeps every token, as llama.cpp's early return does (ch11-facts D2): the float
    // mass before the last tokens can round to 1 or just above it.
    const bool keepAll = p >= 1.0f;
    int count = 0;        // tokens this thread keeps
    float carry = 0.0f;   // the mass of every earlier chunk of this row

    for (unsigned int chunk = 0; chunk < cols; chunk += chunkSize) {
        // Coalesced load of this warp's slice into shared memory.
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int j = chunk + warpSlice + c * warpSize + lane;
            myStage[c * warpSize + lane] = (j < cols) ? rowProbs[j] : 0.0f;
        }
        __syncwarp();

        // Exclusive sequential scan of this lane's run of COARSE_FACTOR tokens.
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            run[c] = threadSum;
            threadSum += myStage[lane * COARSE_FACTOR + c];
        }

        // Block scan of the run totals: warp scans, then warp 0 scans the warp totals.
        float inclusive = warpScan(threadSum);
        float exclusive = __shfl_up_sync(full, inclusive, 1);
        if (lane == 0) {
            exclusive = 0.0f;
        }
        if (lane == warpSize - 1) {
            warpSums_s[warp] = inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            float warpSum = (lane < numWarps) ? warpSums_s[lane] : 0.0f;
            warpSum = warpScan(warpSum);
            if (lane < numWarps) {
                warpSums_s[lane] = warpSum;
            }
        }
        __syncthreads();

        float before = carry + ((warp > 0) ? warpSums_s[warp - 1] : 0.0f) + exclusive;
        carry += warpSums_s[numWarps - 1];

        // Fused: the mass before each token goes to shared memory for a coalesced store; the
        // keep decision is made on that same value, written and counted here. Each lane
        // writes its run's COARSE_FACTOR consecutive bytes, so a warp's byte stores cover one
        // contiguous span.
        const unsigned int first = chunk + warpSlice + lane * COARSE_FACTOR;   // my run's first token
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            float mass = before + run[c];
            myStage[lane * COARSE_FACTOR + c] = mass;
            unsigned int j = first + c;
            if (j < cols) {
                int keepIt = (j == 0 || keepAll || mass < p) ? 1 : 0;
                rowKeep[j] = (unsigned char)keepIt;
                count += keepIt;
            }
        }
        __syncwarp();
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int j = chunk + warpSlice + c * warpSize + lane;
            if (j < cols) {
                rowMass[j] = myStage[c * warpSize + lane];
            }
        }
        __syncthreads();   // every read of warpSums_s is done before the next chunk writes it (lab 11.3)
    }

    // kept[row]: the two-stage reduction of chapter 10 on the threads' counts.
    count = warpSumInt(count);
    if (lane == 0) {
        warpCounts_s[warp] = count;
    }
    __syncthreads();
    if (warp == 0) {
        int total = (lane < numWarps) ? warpCounts_s[lane] : 0;
        total = warpSumInt(total);
        if (lane == 0) {
            kept[blockIdx.x] = total;
        }
    }
}

void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                    float p) {
    if (rows <= 0 || cols <= 0) return;
    TopPMaskKernel<<<rows, BLOCK_DIM>>>(probs, massBefore, keep, kept, (unsigned int)cols, p);
}
