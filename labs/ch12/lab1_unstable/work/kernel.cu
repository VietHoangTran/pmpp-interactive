// Lab 12.1 starter (CUDA; the same file is built for HIP and for the two CPU emulators).
// Three out-of-place unstable filters of unsigned int keys, keep = key < threshold.
// Book: sections 12.2 to 12.4, Figs. 12.2, 12.3 and 12.6 (pp. 290-297).
//
// Contract for all three: input holds n keys; output has room for n; the caller sets
// *outputSize to 0 before every call. Afterwards output[0, *outputSize) must hold exactly the
// kept keys, in any order, and output[*outputSize, n) must be untouched.
//
// Where the book's listings will not do as printed (the README says why):
//   - cond(val) is a placeholder; here it is val < threshold, a kernel argument.
//   - Fig. 12.3 takes __activemask() inside the branch. Write the warp-aggregated kernel with
//     a vote that every lane takes, outside any branch, instead.
//   - Fig. 12.3's mask is a 32-bit unsigned int and its laneIdx() assumes 32 lanes.
//
// On HIP, <cuda/atomic> comes from libhipcxx, which this course's HIP build links only where
// ROCm provides it (notes/today/infra-facts.md B2), so the lab does not rely on it. Where you
// write a cuda::atomic_ref, put the legacy call
// beside it for HIP, which is also relaxed and works on shared and global memory (B1):
//     #if defined(__HIPCC__)
//         ... atomicAdd(address, value) ...
//     #else
//         ... cuda::atomic_ref ... fetch_add(value, cuda::memory_order_relaxed) ...
//     #endif
#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

// One key per thread, BLOCK_DIM threads per block, BLOCK_DIM slots in a private list.
// The build may define it; a multiple of 64 keeps every warp whole on every vendor.
#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif
static_assert(BLOCK_DIM % 64 == 0 && BLOCK_DIM <= 1024, "BLOCK_DIM: a multiple of 64, at most 1024");

// ---- provided: lane masks that work on CUDA and HIP ----
// The type of a warp's lane mask. HIP's _sync intrinsics refuse to compile unless the mask is
// a 64-bit integer, on every AMD target, and __ballot_sync returns unsigned long long there.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Bit counts on a lane mask, chosen by its type (llama.cpp does the same). __popc and __ffs
// take 32 bits on HIP as on CUDA, so a 64-bit mask passed to them would silently lose lanes 32
// to 63. The deleted templates make any other argument type a compile error.
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
// ---- end of the provided part ----

// Fig. 12.2: one atomic per kept key.
__global__ void filterAtomicKernel(const unsigned int* input, unsigned int* output,
                                   unsigned int n, unsigned int threshold,
                                   unsigned int* outputSize) {
    // TODO 1: one thread per key. A thread whose key is in range and below the threshold
    //         takes the next free slot of output from the counter *outputSize, and stores
    //         its key there. Many threads in many blocks share that counter: which atomic
    //         operation, which scope? What does the operation return that you need?
}

// Fig. 12.3, in the documented form: one atomic per warp.
__global__ void filterWarpAggregatedKernel(const unsigned int* input, unsigned int* output,
                                           unsigned int n, unsigned int threshold,
                                           unsigned int* outputSize) {
    // TODO 2: every thread, in range or not, works out whether it keeps a key. Do not
    //         return early: every lane of the warp must reach the vote below.

    // TODO 3: vote. __ballot_sync(mask, predicate), with a full mask of type lane_mask_t,
    //         gives every lane the set of lanes that keep. If that set is empty, the whole
    //         warp can stop here.

    // TODO 4: this thread's lane, from warpSize. Elect the lowest lane that keeps as the
    //         leader. The leader alone reserves room for the whole warp with ONE atomic on
    //         *outputSize: how many slots?

    // TODO 5: the leader hands the start of the reserved range to every lane (a shuffle).

    // TODO 6: a lane that keeps stores its key at start + offset, where offset is the number
    //         of keeping lanes below it. Build the "lanes below me" mask with a shift whose
    //         left operand has type lane_mask_t: what goes wrong with 1 << lane on a
    //         64-lane wavefront?
}

// Fig. 12.6: a private list and counter in shared memory, one public atomic per block.
__global__ void filterPrivatisedKernel(const unsigned int* input, unsigned int* output,
                                       unsigned int n, unsigned int threshold,
                                       unsigned int* outputSize) {
    // TODO 7: declare the block's private list (BLOCK_DIM slots) and private counter in
    //         shared memory. One thread sets the counter to 0. Shared memory starts with
    //         whatever was there. Who must wait for that?

    // TODO 8: as in filterAtomicKernel, but into the private list. Which threads share the
    //         private counter, so which scope does its atomic need?

    // TODO 9: one thread reserves the block's range in the public list, with one
    //         device-scope atomic, and leaves the start where every thread of the block can
    //         read it. What must be true of the private counter before that thread reads
    //         it, and of the start before the others read it?

    // TODO 10: commit: thread k copies private entry k to the public list. Why k, and not
    //          the key this thread filtered?
}

// TODO 11: the three launchers. One thread per key, BLOCK_DIM threads per block; when n is 0
//          there is nothing to launch. The caller zeroes *outputSize.
void launchFilterAtomic(const unsigned int* input, unsigned int* output, unsigned int n,
                        unsigned int threshold, unsigned int* outputSize) {
}

void launchFilterWarpAggregated(const unsigned int* input, unsigned int* output, unsigned int n,
                                unsigned int threshold, unsigned int* outputSize) {
}

void launchFilterPrivatised(const unsigned int* input, unsigned int* output, unsigned int n,
                            unsigned int threshold, unsigned int* outputSize) {
}
