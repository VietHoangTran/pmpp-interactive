// Fig. 11.13 (p. 271) as printed: the coarsened block scan with each thread's run in a
// local array (register tiling). Like Fig. 11.12 it never reads N, so it is right only
// when N is a multiple of COARSE_FACTOR*BLOCK_DIM. blockScan is Fig. 11.9, in
// s114_blockscan_as_printed.cu. Section 11.7's "What registers save" counts its
// shared-memory accesses line by line, and its bank map draws the access on line 16
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
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
    #pragma unroll
    for(unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];
    }
    // Block-wide scan of thread sums
    float threadSum = buffer_r[COARSE_FACTOR - 1];
    threadSum = blockScan(threadSum);
    // Collect thread partial sums
    __shared__ float threadSums[BLOCK_DIM];
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();
    // Add previous thread's partial sums
    float prevPartialSum = (threadIdx.x>0)?threadSums[threadIdx.x - 1]:0.0f;
    #pragma unroll
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prevPartialSum;
    }
    __syncthreads();
    // Write output
    for(unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        output[blockSegment + c*BLOCK_DIM + threadIdx.x] =
                        buffer_s[c*BLOCK_DIM + threadIdx.x];
    }
}
