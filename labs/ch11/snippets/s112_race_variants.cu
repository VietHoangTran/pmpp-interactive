// Broken on purpose: the variants of Figs. 11.3 and 11.5 that the bug hunts of sections
// 11.2 and 11.3 let you switch on. Each one changes one thing in a kernel of
// s112_kogge_stone.cu. The steppers print the same text (app/src/sim/ksblock.ts builds
// it and app/src/sim/ksblock.test.ts checks it against this file), and the same test
// runs every function here on the course's CPU emulator and compares the outputs, thread
// order by thread order, with the model.

#ifndef SEG_SIZE
#define SEG_SIZE 1024
#endif

// Fig. 11.3 without the second barrier (line 14).

__global__ void no_barrier2_scan_kernel(float *input, float *output, unsigned int N) {
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
        if(threadIdx.x >= stride)
            buffer_s[threadIdx.x] = temp;
    }
    if(i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
}

// Fig. 11.3 without the first barrier (line 10).

__global__ void no_barrier1_scan_kernel(float *input, float *output, unsigned int N) {
    __shared__ float buffer_s[SEG_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        buffer_s[threadIdx.x] = input[i];
    } else {
        buffer_s[threadIdx.x] = 0.0f;
    }
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
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

// Fig. 11.3 adding straight into buffer_s, with no temp: the second barrier can then only
// come after the write.

__global__ void direct_add_scan_kernel(float *input, float *output, unsigned int N) {
    __shared__ float buffer_s[SEG_SIZE];
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < N) {
        buffer_s[threadIdx.x] = input[i];
    } else {
        buffer_s[threadIdx.x] = 0.0f;
    }
    for(unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        if(threadIdx.x >= stride)
            buffer_s[threadIdx.x] += buffer_s[threadIdx.x-stride];
        __syncthreads();
    }
    if(i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
}

// Fig. 11.5 without the copy branch (lines 13 to 15).

__global__ void no_copy_double_buffer_scan_kernel(float* input, float* output, unsigned int N) {
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
        }
        float* temp = inBuffer_s;
        inBuffer_s = outBuffer_s;
        outBuffer_s = temp;
    }
    if(i < N) {
        output[i] = inBuffer_s[threadIdx.x];
    }
}
