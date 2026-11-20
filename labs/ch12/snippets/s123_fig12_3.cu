// Fig. 12.3 (p. 292): an unstable filter whose passing lanes share one atomic, through
// __activemask, __ffs, __popc and a shuffle. CUDA only, as on the page: the masks and
// laneIdx() assume 32-lane warps, and HIP rejects the 32-bit mask of __shfl_sync
// (notes/today/ch12-facts.md B1). The kernel keeps the book's lines 01 to 29, so the
// lesson's line numbers are the book's. Shown in section 12.3; its lines are pinned, and
// it is run on the CPU emulator, by app/src/chapters/ch12/S123.snippets.test.ts.
//
// One change to the kernel: line 08 gives j a value. On the page j is uninitialised in
// every lane but the leader, and those lanes hand it to __shfl_sync on line 20; C++ makes
// reading an uninitialised int undefined. When the course's emulator was built with clang
// at -O1 (2026-11-20), the compiler took that licence and every passing lane ran the
// leader's branch: four atomics instead of one, and the counter too high. With j = 0 the
// question does not arise. Whether nvcc does anything similar was not checked.
//
// What the page leaves out, added here so that the file compiles:
//   - cond(): the book's placeholder predicate (p. 291). This one keeps keys below a
//     threshold, as lab 12.1 does.
//   - laneIdx(): chapter 10's helper (p. 239), on the book's 32-lane warp.
//   - <cuda/atomic>, for cuda::atomic_ref.
#include <cuda/atomic>

#define WARP_SIZE 32  // the book's assumption: every warp has 32 lanes
#ifndef FILTER_THRESHOLD
#define FILTER_THRESHOLD 100u
#endif

__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }
__device__ bool cond(unsigned int val) { return val < FILTER_THRESHOLD; }

__global__ void filter_kernel(unsigned int* input, unsigned int* output,
    unsigned int N, unsigned int* outputSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int val = input[i];
        if(cond(val)) {
            unsigned int activeThreads = __activemask();
            unsigned int j = 0;
            // Assign a leader thread
            unsigned int leader = __ffs(activeThreads) - 1;
            if(laneIdx() == leader) {
                // Find how many threads are active
                unsigned int numActive = __popc(activeThreads);
                // Have the leader perform the atomic operation
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    outputSize_ref(*outputSize);
                j = outputSize_ref.fetch_add(numActive, cuda::memory_order_relaxed);
            }
            // Broadcast result to other threads
            j = __shfl_sync(activeThreads, j, leader);
            // Find the position of each active thread in the output
            unsigned int previousThreads = (1 << laneIdx()) - 1;
            unsigned int previousActiveThreads = activeThreads & previousThreads;
            unsigned int offset = __popc(previousActiveThreads);
            // Store the result
            output[j + offset] = val;
        }
    }
}
