// Fig. 10.14 (p. 240), corrected so that one source serves CUDA and HIP. Differences
// from the page: the mask has a type that is 32 bits for CUDA and 64 for HIP; the warp
// width is warpSize, a run-time value, not a literal; the helper uses it too. The mask
// type, warpIdx and warp_reduce are the ones of lab 10.1 (labs/ch10/lab1_sum/solution/
// kernel.cu); app/src/chapters/ch10/S107.snippets.test.ts fails if they drift apart.
// The kernel keeps the book's lines 01 to 19. BLOCK_DIM must equal the launched block
// size, a power of two and at least one warp.
#ifndef BLOCK_DIM
#define BLOCK_DIM 1024
#endif

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

__device__ __inline__ float warp_reduce(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;             // every lane of the warp
    float partialSum = val;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        partialSum += __shfl_down_sync(full, partialSum, stride);
    }
    return partialSum;
}

__global__ void WarpLevelSumReductionKernel(float* input, float* output) {
    __shared__ float input_s[BLOCK_DIM];
    unsigned int t = threadIdx.x;
    input_s[t] = input[t] + input[t + BLOCK_DIM];
    for (unsigned int stride = blockDim.x / 2; stride >= (unsigned int)warpSize; stride /= 2) {
        __syncthreads();
        if (threadIdx.x < stride) {
            input_s[t] += input_s[t + stride];
        }
    }
    __syncthreads();
    if (warpIdx() == 0) {
        float partialSum = input_s[t];
        partialSum = warp_reduce(partialSum);
        if (threadIdx.x == 0) {
            *output = partialSum;
        }
    }
}
