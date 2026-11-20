// Section 10.6's bug hunt, "Where does the barrier go?". One block of BLOCK_DIM threads
// sums 2*BLOCK_DIM values through shared memory. BLOCK_DIM must equal the launched
// blockDim.x and be a power of two; it is 8 here because the hunt runs on 16 values.
//
// SharedMemorySumReductionKernel is Fig. 10.10 (p. 237) with the output parameter its
// body writes, which the printed header lacks. The next four move or remove its barrier
// or change its active test. The last two run a max and then a sum through one scratch
// array, the shape of a softmax (which sums exp(x - max); here the sum is of x - max so
// the arithmetic stays exact). Only SharedMemorySumReductionKernel and MaxThenSumKernel
// are correct; the others are wrong on purpose. app/src/sim/reduction.ts runs each one,
// and app/src/sim/reduction.test.ts pins the line numbers it highlights (line 01 of each
// function is its header).
#define BLOCK_DIM 8

__global__ void SharedMemorySumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

__global__ void BarrierAtBottomKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

__global__ void NoBarrierKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

__global__ void BarrierInsideIfKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        if (threadIdx.x < stride) {
            __syncthreads();
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

__global__ void LessOrEqualKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (threadIdx.x <= stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    if (threadIdx.x == 0) {
        *output = input_s[0];
    }
}

__global__ void MaxThenSumKernel(float* input, float* output) {
    __shared__ float buf_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    float a = input[t], b = input[t + BLOCK_DIM];
    buf_s[t] = a > b ? a : b;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            buf_s[t] = buf_s[t] > buf_s[t + stride] ? buf_s[t] : buf_s[t + stride];
        }
    }
    __syncthreads();
    float m = buf_s[0];
    __syncthreads();
    buf_s[t] = (a - m) + (b - m);
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            buf_s[t] += buf_s[t + stride];
        }
    }
    if (t == 0) {
        *output = buf_s[0];
    }
}

__global__ void MaxThenSumNoBarrierKernel(float* input, float* output) {
    __shared__ float buf_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    float a = input[t], b = input[t + BLOCK_DIM];
    buf_s[t] = a > b ? a : b;
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            buf_s[t] = buf_s[t] > buf_s[t + stride] ? buf_s[t] : buf_s[t + stride];
        }
    }
    __syncthreads();
    float m = buf_s[0];
    buf_s[t] = (a - m) + (b - m);
    for (unsigned int stride = blockDim.x/2; stride >= 1; stride /= 2) {
        __syncthreads();
        if (t < stride) {
            buf_s[t] += buf_s[t + stride];
        }
    }
    if (t == 0) {
        *output = buf_s[0];
    }
}
