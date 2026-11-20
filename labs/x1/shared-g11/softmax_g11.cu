// PROVIDED, do not edit. A copy of labs/ch10/lab3_softmax_argmax/solution/kernel.cu (lab 10.3: row max, softmax, argmax),
// linked into lab X1.3 beside other copies. The only change: helper functions are static, so
// that no two linked files define the same name. app/src/chapters/x1/copies-g11.test.ts checks
// that nothing else differs.
// Lab 10.3 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M4: row-wise max, softmax and argmax over logits.
//
// Book: the two-stage warp-level reduction (section 10.8, Fig. 10.16, p. 243) on the warp
// reduce of Fig. 10.13 (p. 240), the coarsening loop of Fig. 10.20 (p. 247), and exercise 4's
// max operator (p. 250). Softmax and argmax are not in the book: they are ours, built from
// those parts.
//
// Contract (README.md states it in full):
//   x is [rows x cols] float32, row-major, batch first; rows >= 1, cols >= 1; every logit
//   finite. One block of BLOCK_SIZE threads per row.
//   launchRowMax       out[r] = max over c of x[r][c]
//   launchSoftmaxRows  y[r][c] = exp(scale*x[r][c] - m) / sum over c' of exp(scale*x[r][c'] - m),
//                      where m = max over c of scale*x[r][c]
//   launchArgmaxRows   idx[r] = the lowest c with x[r][c] equal to the row's maximum
//
// Where this file differs from the page, and why:
//   - Fig. 10.13 passes the literal 0xffffffff and starts its loop at WARP_SIZE/2. HIP rejects a
//     32-bit mask and AMD warps may have 64 lanes (notes/today/ch10-facts.md B1), so the mask
//     has its own type and the loop starts at warpSize/2, as in lab 10.1.
//   - Fig. 10.16 lets every lane of warp 0 read partialSums_s[t]; only one entry per warp was
//     written. Here the other lanes take the identity (ch10-facts B2; llama.cpp, MLX and
//     PyTorch guard the same read, D1, D3, D4).
//   - Fig. 10.16 needs the result in thread 0 only. Softmax needs it in every thread, so the
//     block reductions leave it in scratch[0] behind a second barrier (MLX does the same, D3).
//   - Fig. 10.20 gives each thread a fixed COARSE_FACTOR and commits across blocks with an
//     atomic. Here one block owns a whole row, so the coarsening loop simply runs to the end
//     of the row (factor ceil(cols / blockDim.x)) and no atomic is needed.
//   - The book's identity for max is -infinity (p. 222). This uses -FLT_MAX, which is finite,
//     so that all three backends share one contract: under Metal's fast math the behaviour
//     of infinity is undefined (ch10-facts C3).
#include <cfloat>
#include <climits>

// The course's CUDA front-end check (labs/pmpp.py) declares no CUDA Math API functions,
// so it is given expf's device declaration here. nvcc, hipcc and the emulator never see it.
#if defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float expf(float x);
#endif

#define BLOCK_SIZE 256                          // threads per block; one block per row
#define MIN_WARP_SIZE 32                        // the narrowest warp any backend here has (ch10-facts B2)
#define MAX_WARPS (BLOCK_SIZE / MIN_WARP_SIZE)  // per-warp slots in shared memory

// Fig. 10.11's first parameter. CUDA: unsigned. HIP: every _sync intrinsic static_asserts
// a 64-bit mask (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
static __device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
static __device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
static __device__ unsigned int warpsInBlock() { return (blockDim.x + warpSize - 1) / warpSize; }

// ---------------------------------------------------------------- operators
// Finite inputs only. On NaN a comparison-based max and CUDA's fmaxf disagree
// (ch10-facts A10); NaN is outside the contract.
static __device__ float maxOp(float a, float b) { return a > b ? a : b; }

// The argmax combine: the larger value wins; equal values go to the lower index. On finite
// values this is commutative and associative, so any tree, any shuffle order and any
// coarsening give the same pair (the property sections 10.4 and 10.9 ask of an operator).
static __device__ void argmaxCombine(float& v, int& i, float otherV, int otherI) {
    if (otherV > v || (otherV == v && otherI < i)) {
        v = otherV;
        i = otherI;
    }
}

// ---------------------------------------------------------------- warp level (Fig. 10.13)
// Every lane calls these; lane 0 returns the warp's result. A lane whose partner lies past
// the end of the warp gets its own value back (ch10-facts A3): harmless for max (max(a, a)
// is a) and, for the sum, those lanes compute values nobody reads.
static __device__ float warpReduceMax(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val = maxOp(val, __shfl_down_sync(full, val, stride));
    }
    return val;
}

static __device__ float warpReduceSum(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

// A shuffle moves one value, so the pair travels as two shuffles, value and index
// separately (MLX does the same, ch10-facts D3).
static __device__ void warpArgmax(float& v, int& i) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        float otherV = __shfl_down_sync(full, v, stride);
        int otherI = __shfl_down_sync(full, i, stride);
        argmaxCombine(v, i, otherV, otherI);
    }
}

// ---------------------------------------------------------------- block level (Fig. 10.16)
// Called by every thread of the block. Returns the block's result to every thread.
// scratch: MAX_WARPS floats of shared memory.
static __device__ float blockReduceMax(float val, float* scratch) {
    val = warpReduceMax(val);                          // stage 1: every warp, in registers
    if (laneIdx() == 0) {
        scratch[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {                              // stage 2: warp 0 reduces the warp results
        val = laneIdx() < warpsInBlock() ? scratch[laneIdx()] : -FLT_MAX;
        val = warpReduceMax(val);
        if (laneIdx() == 0) {
            scratch[0] = val;                          // hand the result to the whole block
        }
    }
    __syncthreads();
    return scratch[0];
}

static __device__ float blockReduceSum(float val, float* scratch) {
    val = warpReduceSum(val);
    if (laneIdx() == 0) {
        scratch[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        val = laneIdx() < warpsInBlock() ? scratch[laneIdx()] : 0.0f;
        val = warpReduceSum(val);
        if (laneIdx() == 0) {
            scratch[0] = val;
        }
    }
    __syncthreads();
    return scratch[0];
}

// ---------------------------------------------------------------- row max
__global__ void rowMaxKernel(const float* x, float* out, int cols) {
    __shared__ float scratch[MAX_WARPS];
    const float* row = x + (size_t)blockIdx.x * cols;   // block b owns row b

    // Coarsening loop (Fig. 10.20): elements blockDim.x apart, so each pass of the block
    // reads one contiguous run (coalesced). No barrier inside.
    float m = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        m = maxOp(m, row[c]);
    }
    m = blockReduceMax(m, scratch);
    if (threadIdx.x == 0) {
        out[blockIdx.x] = m;
    }
}

void launchRowMax(const float* x, float* out, int rows, int cols) {
    rowMaxKernel<<<rows, BLOCK_SIZE>>>(x, out, cols);
}

// ---------------------------------------------------------------- softmax
__global__ void softmaxRowsKernel(const float* x, float* y, int cols, float scale) {
    __shared__ float scratch[MAX_WARPS];
    const float* in = x + (size_t)blockIdx.x * cols;
    float* out = y + (size_t)blockIdx.x * cols;

    // Pass 1: m = max of scale*x. Subtracting it keeps every exponent at or below 0, so no
    // term overflows (exp(90) is beyond float's range; appendix A, pp. 590-591).
    float m = -FLT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        m = maxOp(m, scale * in[c]);
    }
    m = blockReduceMax(m, scratch);

    // Every thread has just read m from scratch[0], and the next reduction writes scratch[0]
    // again (warp 0, lane 0, stage 1). Without this barrier a fast warp 0 can overwrite m
    // before a slow warp has read it. llama.cpp and PyTorch place the same barrier between
    // back-to-back block reductions (ch10-facts D1, D4).
    __syncthreads();

    // Pass 2: s = sum of exp(scale*x - m). Every term is in (0, 1] and the largest is 1,
    // so s >= 1 and the division below is safe.
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        s += expf(scale * in[c] - m);
    }
    s = blockReduceSum(s, scratch);

    // Pass 3: normalise. One division per row, one multiplication per element.
    const float inv = 1.0f / s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        out[c] = expf(scale * in[c] - m) * inv;
    }
}

void launchSoftmaxRows(const float* x, float* y, int rows, int cols, float scale) {
    softmaxRowsKernel<<<rows, BLOCK_SIZE>>>(x, y, cols, scale);
}

// ---------------------------------------------------------------- argmax
__global__ void argmaxRowsKernel(const float* x, int* idx, int cols) {
    __shared__ float vals_s[MAX_WARPS];
    __shared__ int idxs_s[MAX_WARPS];
    const float* row = x + (size_t)blockIdx.x * cols;

    // The identity pair loses to every real pair, a logit of -FLT_MAX included (equal value,
    // and INT_MAX is above every real index).
    float v = -FLT_MAX;
    int i = INT_MAX;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) {
        argmaxCombine(v, i, row[c], c);
    }

    warpArgmax(v, i);
    if (laneIdx() == 0) {
        vals_s[warpIdx()] = v;
        idxs_s[warpIdx()] = i;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        v = -FLT_MAX;
        i = INT_MAX;
        if (laneIdx() < warpsInBlock()) {
            v = vals_s[laneIdx()];
            i = idxs_s[laneIdx()];
        }
        warpArgmax(v, i);
        if (threadIdx.x == 0) {
            idx[blockIdx.x] = i;
        }
    }
}

void launchArgmaxRows(const float* x, int* idx, int rows, int cols) {
    argmaxRowsKernel<<<rows, BLOCK_SIZE>>>(x, idx, cols);
}
