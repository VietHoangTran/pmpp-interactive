// Lab 12.1 reference solution (CUDA; the same file is built for HIP and for the two CPU
// emulators). Three out-of-place unstable filters of unsigned int keys, keep = key < threshold.
//   filterAtomicKernel          book Fig. 12.2 (p. 291): one device-scope atomic per kept key.
//   filterWarpAggregatedKernel  book Fig. 12.3 (p. 292), in the documented form: one atomic
//                               per warp, from a vote taken by every lane.
//   filterPrivatisedKernel      book Fig. 12.6 (p. 296): a private list and counter in shared
//                               memory, one public atomic per block, a coalesced commit.
//
// Contract: input holds n keys; output has room for n; the caller sets *outputSize to 0
// before every call. Afterwards output[0, *outputSize) holds the kept keys in no particular
// order (that is what "unstable" allows), and output[*outputSize, n) is untouched.
//
// Where this file differs from the page:
//   - The book's cond(val) is a placeholder it never defines. Here it is val < threshold,
//     with threshold a kernel argument, so the tests can choose the fraction kept.
//   - Names and const on the input. The book calls every kernel filter_kernel.
//   - Fig. 12.3 reads __activemask() inside the branch and treats it as the lanes that kept
//     a key. NVIDIA's guide now says __activemask() "cannot be used to determine which warp
//     lanes execute a given branch" (notes/today/ch12-facts.md A1). Here every lane, in range
//     or not, votes with __ballot_sync in uniform control flow, which gives exactly the lanes
//     that keep, and the shuffle uses the full mask because every lane executes it.
//   - The mask has its own type, lane_mask_t: unsigned on CUDA and the emulators, unsigned
//     long long on HIP, where every _sync intrinsic rejects a 32-bit mask and a wavefront
//     can be 64 lanes (ch12-facts B1). The bit counts go through typed helpers, and the
//     lanes-below mask is built with a shift of that type, never 1 << lane.
//   - The lane index comes from warpSize. The book's laneIdx() (chapter 10) assumes 32.
//   - Fig. 12.6's shared j is j_s here, so the two j's of the page have different names.
//   - The book says the counter is zeroed before the launch only for Fig. 12.2; here the
//     caller zeroes it for all three.
//   - The book prints no host code. The launchers are ours.
//   - On HIP the atomics are the legacy atomicAdd, not cuda::atomic_ref: see below.

// cuda::atomic_ref comes from <cuda/atomic>. On HIP that header is libhipcxx, which the
// course's HIP build links only where ROCm provides it (notes/today/infra-facts.md B2). So
// the legacy atomicAdd stands in: relaxed, valid on shared and global memory (infra B1).
#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

// One key per thread, BLOCK_DIM threads per block, BLOCK_DIM slots in a private list.
// The build may define it; a multiple of 64 keeps every warp whole on every vendor
// (AMD wavefronts can be 64 lanes, notes/today/ch05-facts.md B5).
#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif
static_assert(BLOCK_DIM % 64 == 0 && BLOCK_DIM <= 1024, "BLOCK_DIM: a multiple of 64, at most 1024");

// region: helpers
// The type of a warp's lane mask. HIP's _sync intrinsics static_assert a 64-bit mask on
// every AMD target, and __ballot_sync returns unsigned long long there (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Bit counts on a lane mask, chosen by its type, as llama.cpp does (ch12-facts D1). __popc
// and __ffs take 32 bits on HIP as on CUDA, so a 64-bit mask passed to them would lose
// lanes 32 to 63 without a word. The deleted template turns any other argument type (an
// unsigned int that holds a HIP ballot, say) into a compile error.
static __device__ __forceinline__ int maskPopc(lane_mask_t m) {
#if defined(__HIPCC__)
    return __popcll(m);
#else
    return __popc(m);
#endif
}
template <typename T> static __device__ __forceinline__ int maskPopc(T m) = delete;

// Position of the lowest set bit, counting from 1; 0 if no bit is set (ch12-facts A2).
static __device__ __forceinline__ int maskFfs(lane_mask_t m) {
#if defined(__HIPCC__)
    return __ffsll(m);
#else
    return __ffs((int)m);       // __ffs takes an int; the bit pattern is unchanged
#endif
}
template <typename T> static __device__ __forceinline__ int maskFfs(T m) = delete;
// endregion

// region: atomic
__global__ void filterAtomicKernel(const unsigned int* input, unsigned int* output,
                                   unsigned int n, unsigned int threshold,
                                   unsigned int* outputSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        unsigned int val = input[i];
        if(val < threshold) {
            // fetch_add returns the old count: that is this key's slot.
#if defined(__HIPCC__)
            unsigned int j = atomicAdd(outputSize, 1u);
#else
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outputSize_ref(*outputSize);
            unsigned int j = outputSize_ref.fetch_add(1, cuda::memory_order_relaxed);
#endif
            output[j] = val;
        }
    }
}
// endregion

// region: warp
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

    // Every lane votes, in range or not: bit k of vote is set if lane k keeps its key.
    const lane_mask_t full = ~(lane_mask_t)0;
    lane_mask_t vote = __ballot_sync(full, keep);
    if(vote == 0) {
        return;                                    // nobody keeps: the whole warp leaves
    }

    unsigned int lane = threadIdx.x % warpSize;
    unsigned int leader = maskFfs(vote) - 1;       // the lowest lane that keeps
    unsigned int j = 0;
    if(lane == leader) {
        unsigned int numKept = maskPopc(vote);     // one atomic for the whole warp
#if defined(__HIPCC__)
        j = atomicAdd(outputSize, numKept);
#else
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> outputSize_ref(*outputSize);
        j = outputSize_ref.fetch_add(numKept, cuda::memory_order_relaxed);
#endif
    }
    j = __shfl_sync(full, j, leader);              // every lane learns where the range starts

    if(keep) {
        lane_mask_t lanesBelow = ((lane_mask_t)1 << lane) - 1;
        unsigned int offset = maskPopc(vote & lanesBelow);   // keeping lanes below this one
        output[j + offset] = val;
    }
}
// endregion

// region: private
__global__ void filterPrivatisedKernel(const unsigned int* input, unsigned int* output,
                                       unsigned int n, unsigned int threshold,
                                       unsigned int* outputSize) {
    // Initialize the private counter
    __shared__ unsigned int output_s[BLOCK_DIM];
    __shared__ unsigned int outputSize_s;
    if(threadIdx.x == 0) {
        outputSize_s = 0;
    }
    __syncthreads();

    // Filter into the private list
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
    __syncthreads();

    // Reserve the block's range in the public list: one atomic per block
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

    // Commit: thread k copies entry k, so neighbouring threads write neighbouring slots
    if(threadIdx.x < outputSize_s) {
        output[j_s + threadIdx.x] = output_s[threadIdx.x];
    }
}
// endregion

// One thread per key, BLOCK_DIM threads per block. The privatised kernel needs blockDim.x
// <= BLOCK_DIM (its private list has BLOCK_DIM slots); the warp-aggregated kernel needs
// whole warps. The caller zeroes *outputSize.
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
