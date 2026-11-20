// A kernel of our own for chapter 9's summary page (the mixed check). It is Fig. 9.10's
// shared-memory privatisation (p. 214) with one change: a thread with no pixel to count
// returns at once. It has one defect on purpose; the page asks the reader to find it.
// app/src/chapters/ch09/Wrap.snippets.test.ts runs it on the course's emulator through
// check/wrap_early_exit_test.cu and compares the bins it gets wrong with the page's model.
#include <cuda/atomic>

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

__global__ void histogramEarlyExit(const unsigned char* image, unsigned int* bins, unsigned int n) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i >= n) return;    // no pixel for this thread: nothing to do

    // Initialize the private bins
    __shared__ unsigned int bins_s[NUM_BINS];
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();

    // Histogram
    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[image[i]]);
    bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
    __syncthreads();

    // Commit to global memory
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if(bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}
