// Lab 10.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// One sum, three kernels. Book: sections 10.6 to 10.10, pp. 236-248.
//
//   (a) SharedMemorySumReductionKernel        Fig. 10.10, p. 237
//   (b) warp_reduce                            Fig. 10.13, p. 240
//       TwoStageWarpLevelSumReductionKernel    Fig. 10.16, p. 243
//   (c) CoarsenedSumReductionKernel            Figs. 10.18 and 10.20, pp. 245 and 247,
//                                              completed, plus exercise 5 (p. 250)
//
// Contract: rows of 2*blockDim.x floats for (a) and (b), one block per row, the sum of
// row b in out[b]. Any n >= 1 (below 2^31) for (c), the sum in *out. Block sizes are
// powers of two from 64 to 1024.
//
// Where this file differs from the page, and why:
//   - Fig. 10.10 has no `output` parameter but writes *output; it is restored here.
//   - The book's BLOCK_DIM (compile time) is blockDim.x (run time) here, so one build
//     serves every block size. Shared arrays are sized for the largest block.
//   - Fig. 10.13 passes the literal 0xffffffff and starts its loop at WARP_SIZE/2 = 16.
//     HIP rejects a 32-bit mask at compile time and AMD warps may have 64 lanes
//     (notes/today/ch10-facts.md B1), so the mask has its own type and the loop starts
//     at warpSize/2.
//   - Fig. 10.16 lets every lane of warp 0 read partialSums_s[t]. Only blockDim.x/warpSize
//     entries were written; the other lanes read padding 0, the identity of +
//     (ch10-facts B2; llama.cpp and PyTorch guard the same read, D1 and D4).
//   - Fig. 10.18 never uses its N and Fig. 10.20 stops at "...": here every load is
//     checked against n and contributes 0 when it is out of range (exercise 5).
//   - COARSE_FACTOR is a run-time argument, so the benchmark can sweep it.
//   - On HIP the commit is the legacy atomicAdd, not cuda::atomic_ref: see (c).

// cuda::atomic_ref for the commit in (c). On HIP, <cuda/atomic> comes from libhipcxx,
// which the HIP build links only where ROCm provides it (infra-facts B2), so this lab
// uses the legacy atomicAdd on HIP and builds either way.
#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (B2)

// Fig. 10.11's first parameter. CUDA: unsigned. HIP: every _sync intrinsic
// static_asserts a 64-bit mask, in wave32 mode too (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// ---------------------------------------------------------------- (a)
__global__ void SharedMemorySumReductionKernel(const float* input, float* output) {
    __shared__ float input_s[MAX_BLOCK_DIM];
    input += 2 * blockDim.x * blockIdx.x;                 // block b reduces row b
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + blockDim.x];        // first tree level, on the load
    for (unsigned int stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();                                  // at the top: also orders the load
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input_s[0];                  // only thread 0 wrote input_s[0] last
    }
}

void launchSharedMemorySum(const float* in, float* out, int rows, int block) {
    SharedMemorySumReductionKernel<<<rows, block>>>(in, out);
}

// ---------------------------------------------------------------- (b)
// Every lane calls it; lane 0 returns the sum of the warp's values. A lane whose
// partner lies past the end of the warp gets its own value back (ch10-facts A3), so the
// upper lanes compute sums nobody reads.
__device__ __inline__ float warp_reduce(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;             // every lane of the warp
    float partialSum = val;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        partialSum += __shfl_down_sync(full, partialSum, stride);
    }
    return partialSum;
}

__global__ void TwoStageWarpLevelSumReductionKernel(const float* input, float* output) {
    input += 2 * blockDim.x * blockIdx.x;                 // block b reduces row b
    unsigned int t = threadIdx.x;

    // Stage 1: every warp reduces its own values in registers.
    float partialSum = input[t] + input[t + blockDim.x];
    partialSum = warp_reduce(partialSum);

    // The one hand-off through shared memory, and the one barrier.
    __shared__ float partialSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    if (laneIdx() == 0) {
        partialSums_s[warpIdx()] = partialSum;
    }
    __syncthreads();

    // Stage 2: warp 0 reduces the per-warp sums. Lanes past the last warp add 0.
    if (warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float partialSum = laneIdx() < warps ? partialSums_s[laneIdx()] : 0.0f;
        partialSum = warp_reduce(partialSum);
        if (threadIdx.x == 0) {
            output[blockIdx.x] = partialSum;
        }
    }
}

void launchTwoStageWarpSum(const float* in, float* out, int rows, int block) {
    TwoStageWarpLevelSumReductionKernel<<<rows, block>>>(in, out);
}

// ---------------------------------------------------------------- (c)
__global__ void CoarsenedSumReductionKernel(const float* input, float* output,
                                            unsigned int N, unsigned int coarse) {
    unsigned int segment = coarse * 2 * blockDim.x * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    // Coarsening loop: 2*coarse elements per thread, blockDim.x apart, so each pass of
    // the block reads one contiguous run. No barrier. Out of range contributes 0.
    float partialSum = i < N ? input[i] : 0.0f;
    for (unsigned int c = 1; c < coarse * 2; ++c) {
        unsigned int idx = i + c * blockDim.x;
        if (idx < N) {
            partialSum += input[idx];
        }
    }

    // The two-stage block reduction of (b). Every thread takes part, in range or not:
    // a thread that returned early would leave a hole in the shuffles (ch10-facts A3).
    partialSum = warp_reduce(partialSum);
    __shared__ float partialSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    if (laneIdx() == 0) {
        partialSums_s[warpIdx()] = partialSum;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float partialSum = laneIdx() < warps ? partialSums_s[laneIdx()] : 0.0f;
        partialSum = warp_reduce(partialSum);
        if (threadIdx.x == 0) {
            // One atomic per block, in the book's spelling (Fig. 10.18, lines 18-20).
            // HIP takes the legacy call, which is also relaxed and device-wide
            // (infra-facts A1, B1).
#if defined(__HIPCC__)
            atomicAdd(output, partialSum);
#else
            cuda::atomic_ref<float, cuda::thread_scope_device> output_ref(*output);
            output_ref.fetch_add(partialSum, cuda::memory_order_relaxed);
#endif
        }
    }
}

void launchCoarsenedSum(const float* in, float* out, int n, int block, int coarse) {
    cudaMemset(out, 0, sizeof(float));                    // the blocks add into *out
    unsigned int perBlock = 2u * (unsigned)coarse * (unsigned)block;
    unsigned int blocks = ((unsigned)n + perBlock - 1) / perBlock;
    CoarsenedSumReductionKernel<<<blocks, block>>>(in, out, (unsigned)n, (unsigned)coarse);
}
