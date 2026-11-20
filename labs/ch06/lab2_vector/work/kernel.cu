// Lab 6.2 starter (CUDA; also built for HIP and for the CPU emulator).
// Book: section 6.3 (pp. 138-139), Fig. 6.12, Exercise 4 (p. 155); section 6.5 (pp. 141-143).
//
// z = x + y, with each thread owning the four consecutive elements 4i .. 4i+3, two ways:
//   vecAdd4Kernel       four elements per instruction (float4 accesses)
//   vecAddCoarseKernel  four elements per thread, one float at a time
// The arrays hold exactly n floats, and n need not be a multiple of 4. The pointers are
// the allocations themselves, so they are 16-byte aligned.
// The provided vecAddKernel (provided/vecadd.cu) is the baseline you are timed against.

#define COARSENING_FACTOR 4

__global__ void vecAdd4Kernel(const float* x, const float* y, float* z, int n) {
    // TODO 1: which chunk of four elements does this thread own?
    // TODO 2: when the whole chunk is inside the arrays, add it with one float4 load
    //         from x, one from y and one float4 store to z. Fig. 6.12 shows how.
    // TODO 3: when only part of the chunk is inside the arrays, a float4 access would
    //         read or write past the end. Handle the elements that are there one at a time.
    // TODO 4: a thread whose chunk starts past the end must touch nothing.
}

void launchVecAdd4(const float* x, const float* y, float* z, int n) {
    // TODO 5: one thread per chunk (count the partial last chunk), 256 threads per block.
}

__global__ void vecAddCoarseKernel(const float* x, const float* y, float* z, int n) {
    // TODO 6: section 6.5's coarsening loop with COARSENING_FACTOR as the factor: this
    //         thread handles that many consecutive elements, each with ordinary float
    //         accesses and its own check against n. Put #pragma unroll on the loop.
}

void launchVecAddCoarse(const float* x, const float* y, float* z, int n) {
    // TODO 7: enough threads for COARSENING_FACTOR elements each, 256 threads per block.
}
