// Lab 9.2, provided (you do not need to edit this file): the benchmark's two
// comparison kernels. They are tested too, so you can trust their counts.
//   refHistogramBasic       book Fig. 9.6 (p. 207): one thread per pixel, one device-scope
//                           atomic add per pixel on the public bins.
//   refHistogramContiguous  book Fig. 9.12 (p. 215): a private histogram in shared memory;
//                           each thread counts coarseFactor CONSECUTIVE pixels
//                           (contiguous partitioning, Fig. 9.11).
//
// Where this file differs from the page:
//   - Names (the book calls every kernel histogram_kernel) and const on the image.
//   - The book's coarsening factor is a compile-time macro, COARSE_FACTOR. Here it is the
//     run-time argument coarseFactor, so one build can be measured at several factors.
//   - The book prints no host code. The launchers are ours, and the caller zeroes bins.
#include <cuda/atomic>

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

__global__ void refHistogramBasic(const unsigned char* image, unsigned int* bins,
                                  unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
        bins_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
}

// region: contiguous
__global__ void refHistogramContiguous(const unsigned char* image, unsigned int* bins,
                                       unsigned int width, unsigned int height,
                                       unsigned int coarseFactor) {
    // Initialize the private bins
    __shared__ unsigned int bins_s[NUM_BINS];
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        bins_s[b] = 0;
    }
    __syncthreads();

    // Histogram: thread t takes the coarseFactor consecutive pixels starting at
    // segment + t*coarseFactor
    unsigned int segment = coarseFactor*blockIdx.x*blockDim.x;
    for(unsigned int c = 0; c < coarseFactor; ++c) {
        unsigned int i = segment + threadIdx.x*coarseFactor + c;
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

// One thread per pixel, 256 threads per block. The caller zeroes bins.
void launchRefHistogramBasic(const unsigned char* image, unsigned int* bins,
                             unsigned int width, unsigned int height) {
    const unsigned int threads = 256;
    unsigned int n = width*height;
    if (n == 0) return;
    refHistogramBasic<<<(n + threads - 1) / threads, threads>>>(image, bins, width, height);
}

// 256 threads per block, each taking coarseFactor pixels: ceil(n / (coarseFactor*256))
// blocks. The caller zeroes bins.
void launchRefHistogramContiguous(const unsigned char* image, unsigned int* bins,
                                  unsigned int width, unsigned int height, unsigned int coarseFactor) {
    const unsigned int threads = 256;
    unsigned int n = width*height;
    if (n == 0 || coarseFactor == 0) return;
    unsigned int perBlock = coarseFactor*threads;
    refHistogramContiguous<<<(n + perBlock - 1) / perBlock, threads>>>(image, bins, width, height, coarseFactor);
}
