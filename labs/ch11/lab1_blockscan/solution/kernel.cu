// Lab 11.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// One block scan, three ways. Book: sections 11.2 to 11.4, pp. 254-265.
//
//   provided  KoggeStoneKernel               Fig. 11.3, p. 256 (../../shared-g11/bs_two_barrier.cu)
//   (a)       KoggeStoneDoubleBufferKernel   Fig. 11.5, p. 260
//   (b)       warpScan                       Fig. 11.8, p. 264
//             blockScan                      Fig. 11.9, p. 264
//             WarpScanKernel                 Fig. 11.10, p. 265
//
// Contract: block b scans its own segment input[b*blockDim.x ... (b+1)*blockDim.x - 1],
// inclusive, with +, independently of every other block. Any n >= 1. A thread past n
// loads 0, the identity of +, takes part in every barrier and shuffle, and stores
// nothing. Block sizes are powers of two from 64 to 1024, chosen at run time.
//
// Where this file differs from the page, and why:
//   - SEG_SIZE, a compile-time constant equal to the block size, is blockDim.x here, so
//     one build serves every block size. The shared buffers are sized for the largest
//     block.
//   - Fig. 11.8 passes the literal 0xffffffff and loops while stride < WARP_SIZE = 32.
//     HIP rejects a 32-bit mask at compile time, and an AMD wavefront may have 64 lanes,
//     where a loop that stops at 16 scans each half of the wavefront on its own
//     (notes/today/ch11-facts.md B1). The mask has its own type and the loop runs to
//     warpSize.
//   - laneIdx() and warpIdx() (Chapter 10, pp. 238-239) use warpSize, not the constant 32.
//   - Fig. 11.9 declares warpSums_s inside blockScan. CUDA allows that, HIP does not
//     document it and Metal refuses it (ch11-facts B1, C2), so the kernel declares the
//     array and passes a pointer. NUM_WARPS is blockDim.x / warpSize, at run time.

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

// ---------------------------------------------------------------- (a)
__global__ void KoggeStoneDoubleBufferKernel(const float* input, float* output, unsigned int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ float buffer1_s[MAX_BLOCK_DIM];
    __shared__ float buffer2_s[MAX_BLOCK_DIM];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;
    inBuffer_s[threadIdx.x] = (i < N) ? input[i] : 0.0f;
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        // One barrier: the writes of the last iteration are visible (true dependence),
        // and every read of the buffer about to be overwritten is done.
        __syncthreads();
        if (threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x] + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];   // finished: carry it over
        }
        float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if (i < N) {
        output[i] = inBuffer_s[threadIdx.x];   // after the swap, the newest values; this
    }                                          // thread wrote this slot itself
}

void launchKoggeStoneDB(const float* in, float* out, int n, int block) {
    unsigned int blocks = ((unsigned int)n + block - 1) / block;
    KoggeStoneDoubleBufferKernel<<<blocks, block>>>(in, out, (unsigned int)n);
}

// ---------------------------------------------------------------- (b)
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

__global__ void WarpScanKernel(const float* input, float* output, unsigned int N) {
    // One float per warp. 1024 threads in warps of 32 (the narrowest warp documented
    // for any backend here) is 32 warps; wider warps need fewer.
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (i < N) ? input[i] : 0.0f;
    val = blockScan(val, warpSums_s);                 // every thread, in range or not
    if (i < N) {
        output[i] = val;
    }
}

void launchWarpScan(const float* in, float* out, int n, int block) {
    unsigned int blocks = ((unsigned int)n + block - 1) / block;
    WarpScanKernel<<<blocks, block>>>(in, out, (unsigned int)n);
}
