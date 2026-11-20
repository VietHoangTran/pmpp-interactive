// Provided baselines for lab 10.1, not edited: the book's first two sum reductions,
// SimpleSumReductionKernel (Fig. 10.5, p. 228) and ConvergentSumReductionKernel
// (Fig. 10.8, p. 233). Lab 10.1 tests them and times its own kernels against them.
//
// Where this file differs from the page: the book launches ONE block of N/2 threads for
// N <= 2048 values and writes *output. Here each block works on its own row, so a launch
// of `rows` blocks reduces `rows` independent rows of 2*blockDim.x values: block b moves
// `input` to the start of row b and writes output[b]. Inside a block every line is the
// book's. Both kernels overwrite their row (they work in place), so a caller that runs
// them twice must refresh the input in between.
//
// Contract: blockDim.x a power of two (the trees halve cleanly only then), rows >= 1.

__global__ void SimpleSumReductionKernel(float* input, float* output) {
    input += 2 * blockDim.x * blockIdx.x;      // ours: block b reduces row b
    unsigned int i = 2 * threadIdx.x;          // thread t owns location 2t
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {       // active threads drift apart: divergence
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input[0];         // the book: *output = input[0];
    }
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    input += 2 * blockDim.x * blockIdx.x;      // ours: block b reduces row b
    unsigned int i = threadIdx.x;              // thread t owns location t
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {            // active threads stay contiguous
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[blockIdx.x] = input[0];         // the book: *output = input[0];
    }
}

// One block of `block` threads per row of 2*block floats; `in` is overwritten.
void launchSimpleSum(float* in, float* out, int rows, int block) {
    SimpleSumReductionKernel<<<rows, block>>>(in, out);
}

void launchConvergentSum(float* in, float* out, int rows, int block) {
    ConvergentSumReductionKernel<<<rows, block>>>(in, out);
}
