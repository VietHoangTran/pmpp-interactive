// Lab 11.2 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Coarsening and register tiling. Book: sections 11.6 and 11.7, pp. 267-272.
//
//   (a) CoarsenedScanKernel       Fig. 11.12, p. 268
//   (b) RegisterTiledScanKernel   Fig. 11.13, p. 271
//       blockScan                 Fig. 11.9, p. 264, as you wrote it in lab 11.1
//
// Contract: block b scans its own segment of COARSE_FACTOR*blockDim.x elements,
// starting at b*COARSE_FACTOR*blockDim.x, inclusive, with +, independently of every other
// block. Any n >= 1. An element at or past n loads 0, the identity of +, and is not
// stored. Block sizes are powers of two from 64 to 1024; COARSE_FACTOR is 1, 2, 4, 8 or
// 16, with COARSE_FACTOR*blockDim.x at most 8192 (see "Shared memory" below).
//
// Where this file differs from the page, and why:
//   - Figs. 11.12 and 11.13 take N but never use it (notes/ch11.md, "Unclear in the
//     PDF"): they are right only when n is a multiple of the segment. Here every global
//     load and store is checked against N, and an element out of range contributes 0.
//   - COARSE_FACTOR is a template parameter, not a macro, so one build tests and times
//     every factor. The launchers pick the instantiation at run time.
//   - BLOCK_DIM is blockDim.x, chosen at run time. The segment buffer, buffer_s
//     [COARSE_FACTOR*BLOCK_DIM] on the page, is dynamic shared memory sized by the launch.
//     A static array sized for the largest block would reserve four times what a
//     256-thread block uses, and the benchmark would measure that instead. threadSums is
//     sized for the largest block.
//   - The text on p. 269 calls the block scan "blockSum"; it is blockScan (Fig. 11.9).
//     As in lab 11.1, it takes the array of warp totals as a pointer.
//
// Shared memory: COARSE_FACTOR*blockDim.x*4 bytes, plus 4 KiB for threadSums and 128 B
// for the warp totals. A block may use 48 KiB without opting in (ch05-facts A3), so
// COARSE_FACTOR*blockDim.x stays at or below 8192: factor 16 only with blocks up to 512.

#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (ch11-facts B1)

// ---------------------------------------------------------------- from lab 11.1
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;   // HIP: 64-bit masks only (infra-facts B3)
#else
typedef unsigned int lane_mask_t;
#endif

__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

__device__ __forceinline__ float warpScan(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = 1; stride < (unsigned int)warpSize; stride *= 2) {
        float leftVal = __shfl_up_sync(full, val, stride);
        if (laneIdx() >= stride) {
            val += leftVal;
        }
    }
    return val;
}

__device__ __forceinline__ float blockScan(float val, float* warpSums_s) {
    val = warpScan(val);
    if (laneIdx() == (unsigned int)warpSize - 1) {
        warpSums_s[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float warpSum = laneIdx() < warps ? warpSums_s[laneIdx()] : 0.0f;
        warpSum = warpScan(warpSum);
        if (laneIdx() < warps) {
            warpSums_s[laneIdx()] = warpSum;
        }
    }
    __syncthreads();
    if (warpIdx() > 0) {
        val += warpSums_s[warpIdx() - 1];
    }
    return val;
}

// ---------------------------------------------------------------- (a)
template <unsigned int COARSE_FACTOR>
__global__ void CoarsenedScanKernel(const float* input, float* output, unsigned int N) {
    extern __shared__ float buffer_s[];                // COARSE_FACTOR*blockDim.x floats
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    unsigned int blockSegment = blockIdx.x * COARSE_FACTOR * blockDim.x;

    // Load the segment chunk by chunk: in pass c, thread t loads element t of chunk c,
    // so a warp reads consecutive addresses (coalesced). Out of range loads 0.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        buffer_s[c * blockDim.x + threadIdx.x] = idx < N ? input[idx] : 0.0f;
    }
    __syncthreads();

    // Thread t owns the run of COARSE_FACTOR elements starting at t*COARSE_FACTOR, which
    // other threads loaded. Scan it sequentially, in place.
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];
    }

    // Scan the run totals across the block, and keep each thread's scanned total.
    float threadSum = buffer_s[threadSegment + COARSE_FACTOR - 1];
    threadSum = blockScan(threadSum, warpSums_s);
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();

    // Add the total of all earlier runs to this run.
    if (threadIdx.x > 0) {
        float prevThreadSum = threadSums[threadIdx.x - 1];
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[threadSegment + c] += prevThreadSum;
        }
    }
    __syncthreads();

    // Store chunk by chunk, coalesced, as the load was. Out of range stores nothing.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * blockDim.x + threadIdx.x];
        }
    }
}

void launchCoarsenedScan(const float* in, float* out, int n, int block, int coarse) {
    unsigned int segment = (unsigned int)coarse * block;
    unsigned int blocks = ((unsigned int)n + segment - 1) / segment;
    size_t shared = segment * sizeof(float);
    switch (coarse) {
        case 1:  CoarsenedScanKernel<1><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 2:  CoarsenedScanKernel<2><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 4:  CoarsenedScanKernel<4><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 8:  CoarsenedScanKernel<8><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 16: CoarsenedScanKernel<16><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        default: break;   // the driver uses only these five factors
    }
}

// ---------------------------------------------------------------- (b)
template <unsigned int COARSE_FACTOR>
__global__ void RegisterTiledScanKernel(const float* input, float* output, unsigned int N) {
    extern __shared__ float buffer_s[];                // COARSE_FACTOR*blockDim.x floats
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    unsigned int blockSegment = blockIdx.x * COARSE_FACTOR * blockDim.x;

    // Coalesced load through shared memory, as in (a). Shared memory is here only to
    // turn the strided per-thread access into contiguous global reads (p. 272).
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        buffer_s[c * blockDim.x + threadIdx.x] = idx < N ? input[idx] : 0.0f;
    }
    __syncthreads();

    // Each element of the run is read from shared memory once; the running sum stays in
    // registers. With every index a constant after unrolling, buffer_r can live in
    // registers (p. 271).
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
#pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];
    }

    float threadSum = buffer_r[COARSE_FACTOR - 1];
    threadSum = blockScan(threadSum, warpSums_s);
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();

    // Add the earlier runs' total and write back to shared memory in one pass.
    float prevPartialSum = threadIdx.x > 0 ? threadSums[threadIdx.x - 1] : 0.0f;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prevPartialSum;
    }
    __syncthreads();

    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * blockDim.x + threadIdx.x];
        }
    }
}

void launchRegisterTiledScan(const float* in, float* out, int n, int block, int coarse) {
    unsigned int segment = (unsigned int)coarse * block;
    unsigned int blocks = ((unsigned int)n + segment - 1) / segment;
    size_t shared = segment * sizeof(float);
    switch (coarse) {
        case 1:  RegisterTiledScanKernel<1><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 2:  RegisterTiledScanKernel<2><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 4:  RegisterTiledScanKernel<4><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 8:  RegisterTiledScanKernel<8><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        case 16: RegisterTiledScanKernel<16><<<blocks, block, shared>>>(in, out, (unsigned int)n); break;
        default: break;
    }
}
