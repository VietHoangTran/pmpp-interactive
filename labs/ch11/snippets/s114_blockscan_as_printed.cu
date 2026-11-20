// Figs. 11.8, 11.9 and 11.10 (pp. 264-265) as printed, with the two helpers they call
// from chapter 10 (pp. 238-239). CUDA only, warps of 32 lanes: under HIP the 32-bit mask
// fails a static_assert that asks for a 64-bit one, and Metal rejects a threadgroup array
// declared in a helper function. Section 11.4 shows these lines in its "as printed" tab
// and its "Port it" hunt points at them by line number (app/src/sim/warpscan.ts,
// PORT_ISSUES; the unit test there checks the numbers). Not a lab.
#define WARP_SIZE 32  // the book's assumption: every warp has 32 lanes
#ifndef BLOCK_DIM
#define BLOCK_DIM 1024  // not printed: the block size the kernel is launched with
#endif
#define NUM_WARPS (BLOCK_DIM/WARP_SIZE)  // not printed: the book says "NUM_WARPS warps"

__device__ unsigned int warpIdx() { return threadIdx.x/WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }

// Fig. 11.8, p. 264

__device__ inline float warpScan(float val) {
    for(unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
        float leftVal = __shfl_up_sync(0xffffffff, val, stride);
        if(laneIdx() >= stride) {
            val += leftVal;
        }
    }
    return val;
}

// Fig. 11.9, p. 264

__device__ inline float blockScan(float val) {
    // Warp-level scan
    val = warpScan(val);
    // Collect warp sums
    __shared__ float warpSums_s[NUM_WARPS];
    if(laneIdx() == WARP_SIZE - 1) {
        warpSums_s[warpIdx()] = val;
    }
    __syncthreads();
    // Scan warp sums
    if(warpIdx() == 0) {
        float warpSum = (threadIdx.x<NUM_WARPS)?warpSums_s[threadIdx.x]:0.0f;
        warpSum = warpScan(warpSum);
        if(threadIdx.x < NUM_WARPS) {
            warpSums_s[threadIdx.x] = warpSum;
        }
    }
    __syncthreads();
    // Add previous warp's scanned sum
    if(warpIdx() > 0) {
        val += warpSums_s[warpIdx() - 1];
    }
    return val;
}

// Fig. 11.10, p. 265

__global__ void scan_kernel(float* input, float* output, unsigned int N) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    float val = (i < N)?input[i]:0.0f;
    val = blockScan(val);
    if(i < N) {
        output[i] = val;
    }
}
