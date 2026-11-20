// Lab 11.5 starter (CUDA; also built for the CPU emulators; HIP unverified).
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
// block; counter is one unsigned int. Nothing is assumed about what any of the three
// hold on entry. An element at or past n loads 0 and is not stored.
//
// Where the book's listings will not do as printed (the lesson page says why):
//   - Fig. 11.18's parameter list is missing a comma (p. 281). Fixed below.
//   - Fig. 11.18 never uses N. Fixed below: every load and store is checked.
//   - Fig. 11.17 declares a __shared__ variable inside the device function, which HIP
//     does not document and Metal refuses (ch11-facts B1, C2). Here the kernel declares
//     it and passes a pointer.
//
// HIP: <cuda/atomic> comes from libhipcxx (ROCm 10.1 and later), which the course's HIP
// build links where ROCm provides it (infra-facts B2; unverified). AMD documents no forward-progress guarantee
// between workgroups (ch11-facts B3).

#include <cuda/atomic>

#define COARSE_FACTOR 4      // elements per thread; the segment is COARSE_FACTOR*blockDim.x
#define MAX_BLOCK_DIM 1024   // the largest block this kernel accepts
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (ch11-facts B1)

// The driver asks for the factor, to size partialSums and flags (one entry per block).
int coarseFactor() { return COARSE_FACTOR; }

// ---------------------------------------------------------------- from labs 11.1 and 11.2
// Provided. Use yours instead if you like; keep the signatures.
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
// Returns, in every thread, the sum of all elements of the blocks with index below
// `bid`, and publishes the sum up to and including this block for block bid + 1.
// `val` is this thread's last scanned element; in the last thread it is the block's
// total. Every thread of the block must call it.
__device__ float interBlockScan(float val, unsigned int bid, float* partialSums, unsigned int* flags,
                                float* prevBlockPartialSum_s) {
    // TODO 1: Fig. 11.17, with the last thread as the block's leader.
    //   a. Block bid waits until flags[bid - 1] is set, then reads partialSums[bid - 1].
    //      The flag is written by a thread of ANOTHER block: what kind of access, at
    //      which thread scope, and with which memory order must the read be, so that
    //      the read of partialSums cannot see a value from before the flag was set?
    //      NVIDIA's message-passing example (ch11 facts A5) is this pattern. Block 0
    //      waits for nobody.
    //   b. It stores this block's running total to partialSums[bid] (a plain store), then
    //      sets flags[bid] so that block bid + 1 cannot see the flag before the total.
    //   c. The rest of the block needs the earlier blocks' sum too.
    //   A wait for a flag that nobody sets never ends: on the emulator it hangs until
    //   pmpp.py gives up (15 minutes). Press Ctrl-C if you see no progress.
    return 0.0f;
}

// ---------------------------------------------------------------- Fig. 11.18
__global__ void SinglePassScanKernel(const float* input, float* output, unsigned int* blockCounter,
                                     float* partialSums, unsigned int* flags, unsigned int N) {
    extern __shared__ float buffer_s[];                // COARSE_FACTOR*blockDim.x floats
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    __shared__ unsigned int bid_s;
    __shared__ float prevBlockPartialSum_s;

    // TODO 2: the dynamic block index (p. 280). Block b may wait only for a block that
    //         has already started. blockIdx.x promises no such thing. One thread claims
    //         the next index from *blockCounter and shares it with the block. Which
    //         memory order does the claim need, given that nothing else depends on the
    //         order of the claims? (The driver checks the counter after the launch.)
    unsigned int bid = blockIdx.x;
    unsigned int blockSegment = COARSE_FACTOR * bid * blockDim.x;

    // Provided from here on: the register-tiled block scan of lab 11.2, with the carry
    // from interBlockScan added before the store.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        buffer_s[c * blockDim.x + threadIdx.x] = idx < N ? input[idx] : 0.0f;
    }
    __syncthreads();

    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
#pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];
    }

    float threadSum = blockScan(buffer_r[COARSE_FACTOR - 1], warpSums_s);
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();

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

    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * blockDim.x + threadIdx.x];
        }
    }
}

// partialSums and flags have one entry per block.
void launchSinglePassScan(const float* in, float* out, float* partialSums, unsigned int* flags,
                          unsigned int* counter, unsigned int n, unsigned int block) {
    unsigned int segment = COARSE_FACTOR * block;
    unsigned int blocks = (n + segment - 1) / segment;
    // TODO 3: the kernel reads the counter and the flags as zero (pp. 278, 280). Nothing
    //         says what they hold now: a previous launch left its values there. Why can
    //         the kernel not reset them itself?
    SinglePassScanKernel<<<blocks, block, segment * sizeof(float)>>>(in, out, counter, partialSums, flags, n);
}
