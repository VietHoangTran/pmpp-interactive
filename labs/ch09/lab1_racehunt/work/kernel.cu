// Lab 9.1 starter (CUDA; the same file is built for HIP and for the two CPU emulators).
// Two 256-bin histograms of an 8-bit grayscale image, written by a colleague:
//   histogramBasic  after book Fig. 9.6 (p. 207): one thread per pixel.
//   histogramPool   after book Fig. 9.9 (p. 212): one private histogram per block in a
//                   pool in global memory, then a commit to the public histogram.
//
// Between them the two kernels have THREE independent defects. The file compiles, and
// on some backends and some images every test passes. Find and fix all three. The two
// launchers at the bottom are correct. kernel.metal has the same three defects.
//
// Start with the CPU emulator, then its race-detection flavour, then your GPU
// (README.md says why in that order).
#include <cuda/atomic>

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

__global__ void histogramBasic(const unsigned char* image, unsigned int* bins,
                               unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        ++bins[b];
    }
}

__global__ void histogramPool(const unsigned char* image, unsigned int* bins, unsigned int* binsPool,
                              unsigned int width, unsigned int height) {
    // This block's private histogram: NUM_BINS counters of the pool.
    unsigned int* binsPriv = &binsPool[blockIdx.x*NUM_BINS];

    // Update the private histogram. Only this block's threads touch binsPriv.
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> binsPriv_ref(binsPriv[b]);
        binsPriv_ref.fetch_add(1, cuda::memory_order_relaxed);
    }

    // Commit the non-zero private bins to the public histogram. NUM_BINS may exceed the
    // block size, so each thread takes every blockDim.x-th bin.
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if(binsPriv[b] > 0) {
            bins[b] += binsPriv[b];
        }
    }
}

// One thread per pixel, threadsPerBlock threads per block. The caller zeroes bins.
void launchHistogramBasic(const unsigned char* image, unsigned int* bins,
                          unsigned int width, unsigned int height, unsigned int threadsPerBlock) {
    unsigned int n = width*height;
    if (n == 0) return;
    unsigned int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    histogramBasic<<<blocks, threadsPerBlock>>>(image, bins, width, height);
}

// As above, plus the pool: ceil(width*height / threadsPerBlock) x NUM_BINS counters,
// one private histogram per block, zeroed by the caller.
void launchHistogramPool(const unsigned char* image, unsigned int* bins, unsigned int* binsPool,
                         unsigned int width, unsigned int height, unsigned int threadsPerBlock) {
    unsigned int n = width*height;
    if (n == 0) return;
    unsigned int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    histogramPool<<<blocks, threadsPerBlock>>>(image, bins, binsPool, width, height);
}
