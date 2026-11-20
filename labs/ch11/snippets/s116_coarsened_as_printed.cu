// Fig. 11.12 (p. 268) as printed: the coarsened block scan, each thread's run kept in
// shared memory. It never reads N and has no bounds checks, so it is right only when N
// is a multiple of COARSE_FACTOR*BLOCK_DIM (notes/ch11.md, "Unclear in the PDF").
// blockScan is Fig. 11.9, in s114_blockscan_as_printed.cu. Section 11.7's "What
// registers save" counts this kernel's shared-memory accesses line by line
// (app/src/sim/coarsescan.ts; the unit test there checks the lines). Not a lab.
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4  // not printed: the factor the text uses as its example (p. 269)
#endif
#ifndef BLOCK_DIM
#define BLOCK_DIM 1024  // not printed
#endif

__device__ float blockScan(float val);  // Fig. 11.9, defined in s114_blockscan_as_printed.cu

__global__ void scan_kernel(float* input, float* output, unsigned int N) {
    unsigned int blockSegment = blockIdx.x*COARSE_FACTOR*BLOCK_DIM;
    // Load data to shared memory
    __shared__ float buffer_s[COARSE_FACTOR*BLOCK_DIM];
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[c*BLOCK_DIM + threadIdx.x] =
                        input[blockSegment + c*BLOCK_DIM + threadIdx.x];
    }
    __syncthreads();
    // Scan thread subsegment
    unsigned int threadSegment = threadIdx.x*COARSE_FACTOR;
    for(unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] += buffer_s[threadSegment + c - 1];
    }
    // Block-wide scan of thread sums
    float threadSum = buffer_s[threadSegment + COARSE_FACTOR - 1];
    threadSum = blockScan(threadSum);
    // Collect thread partial sums
    __shared__ float threadSums[BLOCK_DIM];
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();
    // Add previous thread's partial sums
    if(threadIdx.x > 0) {
        float prevPartialSum = threadSums[threadIdx.x - 1];
        for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            buffer_s[threadSegment + c] += prevPartialSum;
        }
    }
    __syncthreads();
    // Write output
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        output[blockSegment + c*BLOCK_DIM + threadIdx.x] =
                        buffer_s[c*BLOCK_DIM + threadIdx.x];
    }
}
