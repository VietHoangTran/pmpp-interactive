// Lab 9.2 reference solution (CUDA; the same file is built for HIP and for the two CPU
// emulators). Two 256-bin image histograms with a private copy in shared memory:
//   histogramShared  book Fig. 9.14 (p. 217): each thread counts coarseFactor pixels,
//                    interleaved (in iteration c the block reads blockDim.x consecutive
//                    pixels), into the block's shared bins with block-scope atomics; then
//                    the non-zero shared bins are added to the public bins with
//                    device-scope atomics.
//   histogramRuns    book Fig. 9.15 (p. 218): the same, plus a run counter per thread, so
//                    that consecutive equal pixels of one thread cost one atomic.
//
// Where this file differs from the page:
//   - Names (the book calls every kernel histogram_kernel) and const on the image.
//   - The book's coarsening factor is a compile-time macro, COARSE_FACTOR. Here it is the
//     run-time argument coarseFactor, so one build can be measured at several factors.
//   - The book prints no host code. The launchers are ours, and the caller zeroes bins.
#include <cuda/atomic>

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

// region: shared
__global__ void histogramShared(const unsigned char* image, unsigned int* bins,
                                unsigned int width, unsigned int height,
                                unsigned int coarseFactor) {
    // Initialize the private bins
    __shared__ unsigned int bins_s[NUM_BINS];
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();

    // Histogram: in iteration c the block's threads read blockDim.x consecutive pixels
    unsigned int segment = coarseFactor*blockIdx.x*blockDim.x;
    for(unsigned int c = 0; c < coarseFactor; ++c) {
        unsigned int i = segment + c*blockDim.x + threadIdx.x;
        if(i < width*height) {
            unsigned char b = image[i];
            cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
            bins_s_ref.fetch_add(1, cuda::memory_order_relaxed);
        }
    }
    __syncthreads();

    // Commit to global memory
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if(bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}
// endregion

// region: runs
__global__ void histogramRuns(const unsigned char* image, unsigned int* bins,
                              unsigned int width, unsigned int height,
                              unsigned int coarseFactor) {
    // Initialize the private bins
    __shared__ unsigned int bins_s[NUM_BINS];
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();

    // Histogram, one run at a time: b is the value of the current run, bin_r its length
    unsigned int segment = coarseFactor*blockIdx.x*blockDim.x;
    if(segment + threadIdx.x < width*height) {
        unsigned int b = image[segment + threadIdx.x];
        unsigned int bin_r = 1;
        for(unsigned int c = 1; c < coarseFactor; ++c) {
            unsigned int i = segment + c*blockDim.x + threadIdx.x;
            if(i < width*height) {
                unsigned char bNext = image[i];
                if(bNext == b) {
                    ++bin_r;
                } else {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
                    bins_s_ref.fetch_add(bin_r, cuda::memory_order_relaxed);
                    b = bNext;
                    bin_r = 1;
                }
            }
        }
        // The last run has not been added yet
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> bins_s_ref(bins_s[b]);
        bins_s_ref.fetch_add(bin_r, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // Commit to global memory
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if(bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}
// endregion

// 256 threads per block, each taking coarseFactor pixels: ceil(n / (coarseFactor*256))
// blocks. Fewer blocks than one per 256 pixels is the point of coarsening: each block
// pays once to zero and once to commit its private bins (section 9.5).
// The caller zeroes bins.
void launchHistogramShared(const unsigned char* image, unsigned int* bins,
                           unsigned int width, unsigned int height, unsigned int coarseFactor) {
    const unsigned int threads = 256;
    unsigned int n = width*height;
    if (n == 0 || coarseFactor == 0) return;
    unsigned int perBlock = coarseFactor*threads;
    histogramShared<<<(n + perBlock - 1) / perBlock, threads>>>(image, bins, width, height, coarseFactor);
}

void launchHistogramRuns(const unsigned char* image, unsigned int* bins,
                         unsigned int width, unsigned int height, unsigned int coarseFactor) {
    const unsigned int threads = 256;
    unsigned int n = width*height;
    if (n == 0 || coarseFactor == 0) return;
    unsigned int perBlock = coarseFactor*threads;
    histogramRuns<<<(n + perBlock - 1) / perBlock, threads>>>(image, bins, width, height, coarseFactor);
}
