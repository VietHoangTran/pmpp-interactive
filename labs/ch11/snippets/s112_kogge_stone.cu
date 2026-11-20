// Figs. 11.3 and 11.5 (pp. 256 and 260): Kogge-Stone inclusive scan of one segment per
// block, launched with SEG_SIZE threads per block. Sections 11.2 and 11.3 show these two
// functions; app/src/sim/ksblock.ts models them thread by thread and
// app/src/sim/ksblock.test.ts runs them on the course's CPU emulator. Each function's
// lines are numbered as on the page (line 01 is the header).
//
// Differences from the page: both kernels are called scan_kernel in the book, so they
// are renamed here to share one file; line 14 of Fig. 11.3 is printed "__ syncthreads();"
// with a space, which would not compile.

#ifndef SEG_SIZE
#define SEG_SIZE 1024
#endif

__global__ void kogge_stone_scan_kernel(float *input, float *output, unsigned int N) {
    __shared__ float buffer_s[SEG_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        buffer_s[threadIdx.x] = input[i];
    } else {
        buffer_s[threadIdx.x] = 0.0f;
    }
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float temp;
        if(threadIdx.x >= stride)
            temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x-stride];
        __syncthreads();
        if(threadIdx.x >= stride)
            buffer_s[threadIdx.x] = temp;
    }
    if(i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
}

__global__ void double_buffer_scan_kernel(float* input, float* output, unsigned int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    __shared__ float buffer1_s[SEG_SIZE];
    __shared__ float buffer2_s[SEG_SIZE];
    float* inBuffer_s = buffer1_s;
    float* outBuffer_s = buffer2_s;
    inBuffer_s[threadIdx.x] = (i < N)?input[i]:0.0f;
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if(threadIdx.x >= stride) {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x]
                                       + inBuffer_s[threadIdx.x - stride];
        } else {
            outBuffer_s[threadIdx.x] = inBuffer_s[threadIdx.x];
        }
        float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if(i < N) {
        output[i] = inBuffer_s[threadIdx.x];
    }
}
