// Lab 9.1 reference solution: the starter's three defects, fixed.
// Two 256-bin histograms of an 8-bit grayscale image, one bin per pixel value:
//   histogramBasic  book Fig. 9.6 (p. 207): one thread per pixel, one device-scope atomic
//                   add per pixel on the public bins.
//   histogramPool   book Fig. 9.9 (p. 212): one private histogram per block in a pool in
//                   global memory, block-scope atomics on it, a barrier, then a commit of
//                   the non-zero private bins to the public bins with device-scope atomics.
//
// Where this file differs from the page:
//   - Names. The book calls both kernels histogram_kernel and the pool and the private copy
//     bins_pool and bins_priv; here they are histogramBasic, histogramPool, binsPool and
//     binsPriv. The image pointer is const.
//   - The book prints no host code and does not say who zeroes bins and the pool
//     (notes/ch09.md, "Unclear in the PDF"). Here the caller zeroes both, and the
//     launchers at the bottom are ours.
#include <cuda/atomic>

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

// D1. The starter incremented the public bin with a plain `++bins[b];`. That is a read,
// an add and a write; two threads that read the same old count both write old + 1, and
// one pixel is lost. Threads of every block update the same bins, so the atomic needs
// device scope (Fig. 9.6, p. 208).
// region: basic
__global__ void histogramBasic(const unsigned char* image, unsigned int* bins,
                               unsigned int width, unsigned int height) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
        bins_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
}
// endregion

// D2. The starter went straight from the update to the commit. A thread then read
// binsPriv[b] while other threads of its block were still adding to it, and committed a
// count that was not final. The barrier makes every update of the block happen before
// any commit (p. 213).
// D3. The starter committed with a plain `bins[b] += binsPriv[b];`. Only one thread per
// block touches a given public bin, but every block commits to the same 256 bins, so two
// blocks can read the same old count. Device scope, because the threads that share
// bins[b] are in different blocks. Block scope there would be a data race under CUDA's
// memory model even where it gives the right numbers (notes/today/ch09-facts.md A2).
// The private update keeps block scope: only this block's threads touch binsPriv.
// region: pool
__global__ void histogramPool(const unsigned char* image, unsigned int* bins, unsigned int* binsPool,
                              unsigned int width, unsigned int height) {
    // This block's private histogram: NUM_BINS counters of the pool.
    unsigned int* binsPriv = &binsPool[blockIdx.x*NUM_BINS];

    // Update the private histogram.
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < width*height) {
        unsigned char b = image[i];
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> binsPriv_ref(binsPriv[b]);
        binsPriv_ref.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();

    // Commit the non-zero private bins to the public histogram. NUM_BINS may exceed the
    // block size, so each thread takes every blockDim.x-th bin.
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if(binsPriv[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> bins_ref(bins[b]);
            bins_ref.fetch_add(binsPriv[b], cuda::memory_order_relaxed);
        }
    }
}
// endregion

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
