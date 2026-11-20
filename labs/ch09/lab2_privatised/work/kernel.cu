// Lab 9.2 starter (CUDA; the same file is built for HIP and for the two CPU emulators).
// Two 256-bin image histograms with a private copy of the bins in shared memory.
// Book: sections 9.4 to 9.6, Figs. 9.10, 9.14 and 9.15 (pp. 213-218).
//
// Contract for both kernels: image holds width*height pixels, one byte each; bins holds
// NUM_BINS counters that the caller has zeroed; each thread counts coarseFactor pixels.
// The book makes the coarsening factor a compile-time macro (COARSE_FACTOR); here it is a
// run-time argument, so one build can be measured at several factors.
// The launchers at the bottom are written for you: 256 threads per block. Do not let the
// kernels depend on that: write them for any block size, as the book does.
#include <cuda/atomic>

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

// Fig. 9.14: a private histogram in shared memory, coarseFactor pixels per thread,
// interleaved partitioning.
__global__ void histogramShared(const unsigned char* image, unsigned int* bins,
                                unsigned int width, unsigned int height,
                                unsigned int coarseFactor) {
    // TODO 1: declare the block's private histogram in shared memory and set every bin
    //         to 0. Shared memory starts with whatever was there before.

    // TODO 2: make sure no thread starts counting before every bin is 0.

    // TODO 3: count this thread's coarseFactor pixels into the private histogram. In
    //         iteration c the block's threads together read blockDim.x consecutive
    //         pixels (interleaved partitioning, Fig. 9.13). Which threads can update the
    //         same private bin, and so which scope does the atomic need?

    // TODO 4: wait until the whole block has finished counting.

    // TODO 5: add each non-zero private bin to the public histogram. Which threads can
    //         update the same public bin now, and so which scope does that atomic need?
}

// Fig. 9.15: histogramShared plus a run counter per thread. Consecutive pixels of one
// thread that have the same value are counted in a register, and the count is added to
// the private histogram with one atomic when the value changes.
__global__ void histogramRuns(const unsigned char* image, unsigned int* bins,
                              unsigned int width, unsigned int height,
                              unsigned int coarseFactor) {
    // TODO 1 and 2: as in histogramShared.

    // TODO 3: if this thread's first pixel exists, start a run: its value, and a length
    //         of 1. Then walk the thread's later pixels with the same interleaved
    //         indexing. A pixel with the run's value makes the run longer; any other value
    //         adds the run to the private histogram and starts a new run. Is every run
    //         in the private histogram by the time the walk ends?

    // TODO 4 and 5: as in histogramShared.
}

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
