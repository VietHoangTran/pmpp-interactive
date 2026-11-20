// Figs. 10.5 and 10.8 (pp. 228 and 233) as printed: one block of N/2 threads sums N
// values, at most 2048, in place in global memory. The steppers of sections 10.3 and
// 10.4 show these two functions, app/src/sim/reduction.ts runs them, and
// app/src/sim/reduction.test.ts pins the line numbers the steppers highlight. Each
// function's lines are numbered as on the page (line 01 is the header).

__global__ void SimpleSumReductionKernel(float* input, float* output) {
    unsigned int i = 2*threadIdx.x;
    for (unsigned int stride = 1; stride <= blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}

__global__ void ConvergentSumReductionKernel(float* input, float* output) {
    unsigned int i = threadIdx.x;
    for (unsigned int stride = blockDim.x; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input[0];
    }
}
