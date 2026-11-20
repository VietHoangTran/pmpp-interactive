// Book exercise 7 (p. 286; the task is set on p. 285): a block-level Brent-Kung inclusive
// scan. The answer is ours (the book prints none), built from Fig. 11.19 (p. 283). Section
// 11.10 shows this file; app/src/sim/prefixnet.ts models it (thread t performs the t-th
// adder of each step) and app/src/sim/prefixnet.test.ts runs it on the course's CPU
// emulator against that model.
//
// Each block scans its own segment of SEG_SIZE elements and is launched with SEG_SIZE/2
// threads: the widest step, the first of the reduction tree, has SEG_SIZE/2 additions.
// One barrier per loop iteration is enough because, within one step, no position written
// by one thread is read by another (the reduction tree writes positions 2*stride*k - 1 and
// reads those minus stride; the reverse tree writes those plus stride).

#ifndef SEG_SIZE
#define SEG_SIZE 2048
#endif

__global__ void brent_kung_scan_kernel(float *input, float *output, unsigned int N) {
    __shared__ float buffer_s[SEG_SIZE];
    unsigned int i = 2*blockIdx.x*blockDim.x + threadIdx.x;
    buffer_s[threadIdx.x] = (i < N)?input[i]:0.0f;
    buffer_s[threadIdx.x + blockDim.x] = (i + blockDim.x < N)?input[i + blockDim.x]:0.0f;
    // Reduction tree: strides 1, 2, 4, ..., SEG_SIZE/2
    for(unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1)*2*stride - 1;
        if(idx < SEG_SIZE) {
            buffer_s[idx] += buffer_s[idx - stride];
        }
    }
    // Reverse tree: strides SEG_SIZE/4, ..., 2, 1
    for(unsigned int stride = SEG_SIZE/4; stride > 0; stride /= 2) {
        __syncthreads();
        unsigned int idx = (threadIdx.x + 1)*2*stride - 1;
        if(idx + stride < SEG_SIZE) {
            buffer_s[idx + stride] += buffer_s[idx];
        }
    }
    __syncthreads();
    if(i < N) {
        output[i] = buffer_s[threadIdx.x];
    }
    if(i + blockDim.x < N) {
        output[i + blockDim.x] = buffer_s[threadIdx.x + blockDim.x];
    }
}
