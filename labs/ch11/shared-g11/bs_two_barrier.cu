// Provided for lab 11.1, not edited: the book's first block scan, Fig. 11.3 (p. 256).
// Kogge-Stone in one shared buffer, two barriers per iteration. It is the baseline the
// lab's other kernels are timed against.
//
// Contract (the same as lab 11.1's kernels): block b scans its own segment of blockDim.x
// elements, inclusive, with +. Threads past N load 0, the identity of +, and store
// nothing. Block sizes are powers of two from 64 to 1024.
//
// Where this file differs from the page:
//   - SEG_SIZE, a compile-time constant equal to the block size, is blockDim.x here, so
//     one build serves every block size. buffer_s is sized for the largest block.
//   - The page prints line 14 as `__ syncthreads();`, with a space (notes/ch11.md,
//     "Unclear in the PDF").
//   - The book names every scan kernel scan_kernel; this one is KoggeStoneKernel, and it
//     takes a const input pointer.

#define BS_MAX_BLOCK_DIM 1024

__global__ void KoggeStoneKernel(const float* input, float* output, unsigned int N) {
    __shared__ float buffer_s[BS_MAX_BLOCK_DIM];
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        buffer_s[threadIdx.x] = input[i];
    } else {
        buffer_s[threadIdx.x] = 0.0f;
    }
    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();                    // barrier 1: last iteration's writes are done
        float temp = 0.0f;
        if (threadIdx.x >= stride) {
            temp = buffer_s[threadIdx.x] + buffer_s[threadIdx.x - stride];
        }
        __syncthreads();                    // barrier 2: this iteration's reads are done
        if (threadIdx.x >= stride) {
            buffer_s[threadIdx.x] = temp;
        }
    }
    if (i < N) {
        output[i] = buffer_s[threadIdx.x];  // the slot this thread wrote last
    }
}

void launchKoggeStone(const float* in, float* out, int n, int block) {
    unsigned int blocks = ((unsigned int)n + block - 1) / block;
    KoggeStoneKernel<<<blocks, block>>>(in, out, (unsigned int)n);
}
