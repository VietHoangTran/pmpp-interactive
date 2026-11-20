// Fig. 12.4 (p. 294): Fig. 12.3 rewritten with a coalesced group from cooperative groups.
// The kernel keeps the book's lines 01 to 26, so the lesson's line numbers are the book's.
// Shown in section 12.3; its lines are pinned, and it is run on the CPU emulator, by
// app/src/chapters/ch12/S123.snippets.test.ts.
//
// Differences from the page:
//   - The page shows no header and no namespace (notes/today/ch12-facts.md A5). NVIDIA's
//     examples write #include <cooperative_groups.h> and namespace cg = cooperative_groups;
//     HIP's header is <hip/hip_cooperative_groups.h> (infra-facts B4), chosen below.
//   - Line 07 reaches the type and the function through cg::.
//   - Line 08 gives j a value, for the reason given in s123_fig12_3.cu: lanes other than
//     rank 0 pass j to shfl on line 19 before anything has been stored in it.
//   - Line 12 calls num_threads(). The book's size() still compiles: NVIDIA now lists it
//     as a legacy alias of num_threads() (A5).
//   - cond() and <cuda/atomic> as in s123_fig12_3.cu.
// Not built for HIP: there <cuda/atomic> needs libhipcxx, which the course's HIP build links
// only where ROCm provides it (infra-facts B2).
#if defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include <cuda/atomic>

namespace cg = cooperative_groups;

#ifndef FILTER_THRESHOLD
#define FILTER_THRESHOLD 100u
#endif

__device__ bool cond(unsigned int val) { return val < FILTER_THRESHOLD; }

__global__ void filter_kernel(unsigned int* input, unsigned int* output,
    unsigned int N, unsigned int* outputSize) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        unsigned int val = input[i];
        if(cond(val)) {
            cg::coalesced_group activeThreads = cg::coalesced_threads();
            unsigned int j = 0;
            // Assign a leader thread
            if(activeThreads.thread_rank() == 0) {
                // Find how many threads are active
                unsigned int numActive = activeThreads.num_threads();
                // Have the leader perform the atomic operation
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    outputSize_ref(*outputSize);
                j = outputSize_ref.fetch_add(numActive, cuda::memory_order_relaxed);
            }
            // Broadcast result to other threads
            j = activeThreads.shfl(j, 0);
            // Find the position of each active thread in the output
            unsigned int offset = activeThreads.thread_rank();
            // Store the result
            output[j + offset] = val;
        }
    }
}
