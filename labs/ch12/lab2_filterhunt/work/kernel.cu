// Lab 12.2 starter: the three unstable filters of lab 12.1 (book Figs. 12.2, 12.3 and 12.6,
// pp. 291-296), keep = key < threshold. CUDA; the same file is built for HIP and for the two
// CPU emulators.
//
// Contract for all three: input holds n keys; output has room for n; the caller sets
// *outputSize to 0 before every call. Afterwards output[0, *outputSize) must hold exactly the
// kept keys, in any order, and output[*outputSize, n) must be untouched.
//
// Between them, the three kernels have THREE independent defects. The file compiles. Find
// and fix all three. Run it on the CPU emulator first (README, "Method").
//
// Not defects: the mask type lane_mask_t and the typed helpers below, the vote that every
// lane takes before any thread returns, the lane index from warpSize, and the legacy
// atomicAdd under __HIPCC__ (lab 12.1's README says why for each).
#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif
static_assert(BLOCK_DIM % 64 == 0 && BLOCK_DIM <= 1024, "BLOCK_DIM: a multiple of 64, at most 1024");

// The type of a warp's lane mask: 64 bits on HIP, 32 on CUDA and the emulators.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Bit counts chosen by the mask's type; any other argument type is a compile error.
static __device__ __forceinline__ int maskPopc(lane_mask_t m) {   // number of set bits
#if defined(__HIPCC__)
    return __popcll(m);
#else
    return __popc(m);
#endif
}
template <typename T> static __device__ __forceinline__ int maskPopc(T m) = delete;

static __device__ __forceinline__ int maskFfs(lane_mask_t m) {    // lowest set bit, from 1; 0 if none
#if defined(__HIPCC__)
    return __ffsll(m);
#else
    return __ffs((int)m);
#endif
}
template <typename T> static __device__ __forceinline__ int maskFfs(T m) = delete;

// Fig. 12.2: one counter update per kept key.
__global__ void filterAtomicKernel(const unsigned int* input, unsigned int* output,
                                   unsigned int n, unsigned int threshold,
                                   unsigned int* outputSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        unsigned int val = input[i];
        if(val < threshold) {
            unsigned int j = *outputSize;          // the next free slot
            *outputSize = j + 1;
            output[j] = val;
        }
    }
}

// Fig. 12.3, with a vote that every lane takes: one atomic per warp.
__global__ void filterWarpAggregatedKernel(const unsigned int* input, unsigned int* output,
                                           unsigned int n, unsigned int threshold,
                                           unsigned int* outputSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int val = 0;
    bool keep = false;
    if(i < n) {
        val = input[i];
        keep = val < threshold;
    }

    const lane_mask_t full = ~(lane_mask_t)0;
    lane_mask_t vote = __ballot_sync(full, keep);
    if(vote == 0) {
        return;                                    // nobody keeps: the whole warp leaves
    }

    unsigned int lane = threadIdx.x % warpSize;
    unsigned int leader = maskFfs(vote) - 1;       // the lowest lane that keeps
    unsigned int j = 0;
    if(lane == leader) {
        unsigned int numKept = maskPopc(vote);
#if defined(__HIPCC__)
        j = atomicAdd(outputSize, numKept);
#else
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outputSize_ref(*outputSize);
        j = outputSize_ref.fetch_add(numKept, cuda::memory_order_relaxed);
#endif
    }
    j = __shfl_sync(full, j, leader);

    if(keep) {
        lane_mask_t lanesBelow = ((lane_mask_t)2 << lane) - 1;
        unsigned int offset = maskPopc(vote & lanesBelow);
        output[j + offset] = val;
    }
}

// Fig. 12.6: a private list and counter in shared memory, one public atomic per block.
__global__ void filterPrivatisedKernel(const unsigned int* input, unsigned int* output,
                                       unsigned int n, unsigned int threshold,
                                       unsigned int* outputSize) {
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    if(threadIdx.x == 0) {
        outputSize_s = 0;
    }
    __syncthreads();

    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        unsigned int val = input[i];
        if(val < threshold) {
#if defined(__HIPCC__)
            unsigned int j = atomicAdd(&outputSize_s, 1u);
#else
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> outputSize_s_ref(outputSize_s);
            unsigned int j = outputSize_s_ref.fetch_add(1, cuda::memory_order_relaxed);
#endif
            output_s[j] = val;
        }
    }

    __shared__ unsigned int j_s;
    if(threadIdx.x == 0) {
#if defined(__HIPCC__)
        j_s = atomicAdd(outputSize, outputSize_s);
#else
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outputSize_ref(*outputSize);
        j_s = outputSize_ref.fetch_add(outputSize_s, cuda::memory_order_relaxed);
#endif
    }
    __syncthreads();

    if(threadIdx.x < outputSize_s) {
        output[j_s + threadIdx.x] = output_s[threadIdx.x];
    }
}

// One thread per key, BLOCK_DIM threads per block. These are correct.
void launchFilterAtomic(const unsigned int* input, unsigned int* output, unsigned int n,
                        unsigned int threshold, unsigned int* outputSize) {
    if(n == 0) return;
    filterAtomicKernel<<<(n + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(input, output, n, threshold, outputSize);
}

void launchFilterWarpAggregated(const unsigned int* input, unsigned int* output, unsigned int n,
                                unsigned int threshold, unsigned int* outputSize) {
    if(n == 0) return;
    filterWarpAggregatedKernel<<<(n + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(input, output, n, threshold, outputSize);
}

void launchFilterPrivatised(const unsigned int* input, unsigned int* output, unsigned int n,
                            unsigned int threshold, unsigned int* outputSize) {
    if(n == 0) return;
    filterPrivatisedKernel<<<(n + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM>>>(input, output, n, threshold, outputSize);
}
