// Lab 11.5 reference solution (CUDA; also built for the CPU emulators; HIP unverified).
// Single-pass scan with single lookback. Book: section 11.9, pp. 276-282.
//
//   interBlockScan          Fig. 11.17, p. 278
//   SinglePassScanKernel    Fig. 11.18, p. 281
//   launchSinglePassScan
//
// No Metal version: MSL documents no forward progress between threadgroups, and an
// Apple engineer called the chained-scan assumptions undefined behaviour (ch11-facts
// C5). Lab 11.4's multi-dispatch scan is the Metal global scan.
//
// Contract: an inclusive scan with + of all n elements, 1 <= n < 2^31, in one launch.
// One block of `block` threads scans a segment of COARSE_FACTOR*block elements. The
// block size is a power of two from 64 to 1024. partialSums and flags hold one entry per
// block; counter is one unsigned int. An element at or past n loads 0 and is not stored.
//
// Where this file differs from the page, and why:
//   - Fig. 11.18's parameter list has no comma between `unsigned int* flags` and
//     `unsigned int N` (p. 281; notes/ch11.md, "Unclear in the PDF"). Restored.
//   - Fig. 11.18 never uses N and has no bounds checks. Every global load and store is
//     checked against N here; an element out of range contributes 0.
//   - Fig. 11.17 reads the flag with fetch_add(0, acquire) and sets it with
//     fetch_add(1, release). NVIDIA's own message-passing example uses
//     load(memory_order_acquire) and store(1, memory_order_release) (ch11-facts A5): a
//     load is the read-only form, and nothing here needs a read-modify-write. The
//     ordering is the same.
//   - Fig. 11.17 declares prevBlockPartialSum_s inside the device function. HIP does not
//     document that and Metal refuses it (ch11-facts B1, C2), so the kernel declares it
//     and passes a pointer, as lab 11.1 does with the warp totals.
//   - Fig. 11.18 mixes blockDim.x (line 12) and BLOCK_DIM (line 14). Here the block size
//     is blockDim.x throughout, chosen at run time, and the segment buffer is dynamic
//     shared memory, as in lab 11.2.
//   - The flags must start at zero (implied on p. 278) and the counter is
//     "zero-initialized" (p. 280). launchSinglePassScan zeroes both with cudaMemset
//     before every launch: the kernel cannot do it, because no block knows when every
//     other block has read them (ch11-facts A5).
//
// HIP: <cuda/atomic> comes from libhipcxx (ROCm 10.1 and later), which the course's HIP
// build links where ROCm provides it (infra-facts B2; unverified). AMD documents no forward-progress guarantee
// between workgroups (ch11-facts B3). This file is unverified on HIP.

#include <cuda/atomic>

#define COARSE_FACTOR 4      // elements per thread; the segment is COARSE_FACTOR*blockDim.x
#define MAX_BLOCK_DIM 1024   // the largest block this kernel accepts
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (ch11-facts B1)

// The driver asks for the factor, to size partialSums and flags (one entry per block).
int coarseFactor() { return COARSE_FACTOR; }

// ---------------------------------------------------------------- from labs 11.1 and 11.2
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

// ---------------------------------------------------------------- Fig. 11.17
// The leader (the last thread, which holds the block's total in `val`) waits until the
// block with index bid - 1 has published the sum of everything before and including it,
// publishes the same for this block, and shares the earlier sum with the whole block.
// Every thread of the block must call it (it ends with a barrier).
__device__ float interBlockScan(float val, unsigned int bid, float* partialSums, unsigned int* flags,
                                float* prevBlockPartialSum_s) {
    if (threadIdx.x == blockDim.x - 1) {
        if (bid > 0) {
            // Device scope: the flag is written by a thread of another block (ch11-facts
            // A5). Acquire: the load of partialSums[bid - 1] below cannot move before it.
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> prevBlockFlag_ref(flags[bid - 1]);
            while (prevBlockFlag_ref.load(cuda::memory_order_acquire) == 0) {
            }
            *prevBlockPartialSum_s = partialSums[bid - 1];
        } else {
            *prevBlockPartialSum_s = 0.0f;
        }
        partialSums[bid] = *prevBlockPartialSum_s + val;
        // Release: the store to partialSums[bid] above cannot move after the flag.
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> myBlockFlag_ref(flags[bid]);
        myBlockFlag_ref.store(1, cuda::memory_order_release);
    }
    __syncthreads();   // the leader's shared write, visible to the block
    return *prevBlockPartialSum_s;
}

// ---------------------------------------------------------------- Fig. 11.18
__global__ void SinglePassScanKernel(const float* input, float* output, unsigned int* blockCounter,
                                     float* partialSums, unsigned int* flags, unsigned int N) {
    extern __shared__ float buffer_s[];                // COARSE_FACTOR*blockDim.x floats
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    __shared__ unsigned int bid_s;
    __shared__ float prevBlockPartialSum_s;

    // Dynamic block index (p. 280): blocks claim indices in the order they start, so the
    // block this one waits for has already started. Relaxed: no other memory access
    // depends on the order of these increments.
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> blockCounter_ref(*blockCounter);
        bid_s = blockCounter_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    unsigned int bid = bid_s;
    unsigned int blockSegment = COARSE_FACTOR * bid * blockDim.x;

    // Coalesced load, chunk by chunk. Out of range loads 0.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        buffer_s[c * blockDim.x + threadIdx.x] = idx < N ? input[idx] : 0.0f;
    }
    __syncthreads();

    // Register-tiled run scan (Fig. 11.13).
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
#pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];
    }

    // Block scan of the run totals.
    float threadSum = blockScan(buffer_r[COARSE_FACTOR - 1], warpSums_s);
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();

    // Add the earlier runs' total; the result stays in registers this time.
    float prevPartialSum = threadIdx.x > 0 ? threadSums[threadIdx.x - 1] : 0.0f;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_r[c] += prevPartialSum;
    }

    // The last thread's last element is now the block's total.
    float prevBlockPartialSum = interBlockScan(buffer_r[COARSE_FACTOR - 1], bid, partialSums, flags,
                                               &prevBlockPartialSum_s);

#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prevBlockPartialSum;
    }
    __syncthreads();

    // Coalesced store. Out of range stores nothing.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * blockDim.x + threadIdx.x];
        }
    }
}

// partialSums and flags have one entry per block. The launch reads the counter and the
// flags as zero, so they are zeroed here, before every launch.
void launchSinglePassScan(const float* in, float* out, float* partialSums, unsigned int* flags,
                          unsigned int* counter, unsigned int n, unsigned int block) {
    unsigned int segment = COARSE_FACTOR * block;
    unsigned int blocks = (n + segment - 1) / segment;
    cudaMemset(counter, 0, sizeof(unsigned int));
    cudaMemset(flags, 0, blocks * sizeof(unsigned int));
    SinglePassScanKernel<<<blocks, block, segment * sizeof(float)>>>(in, out, counter, partialSums, flags, n);
}
