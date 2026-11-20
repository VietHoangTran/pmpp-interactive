// A kernel of our own for chapter 10's summary page (the mixed check). It centres each
// row of a row-major [rows x cols] matrix, y = x - mean(x), the first step of LayerNorm:
// one block per row, blockDim.x a whole number of warps and at most 1,024. It has one
// defect on purpose; the page asks the reader to find it.
// The mask type, warpIdx, laneIdx and warp_reduce are lab 10.1's (labs/ch10/lab1_sum/
// solution/kernel.cu). app/src/chapters/ch10/Wrap.snippets.test.ts fails if they drift
// apart, and runs this kernel on the course's emulator through check/wrap_centre_test.cu.
#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (B2)

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

__device__ __inline__ float warp_reduce(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;             // every lane of the warp
    float partialSum = val;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        partialSum += __shfl_down_sync(full, partialSum, stride);
    }
    return partialSum;
}

__global__ void centreRows(const float* x, float* y, int cols) {
    const float* row = x + blockIdx.x * cols;
    float* out = y + blockIdx.x * cols;

    float s = 0.0f;                                       // coarsened and coalesced
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += row[c];

    s = warp_reduce(s);                                   // stage 1: every warp
    __shared__ float partial_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    if (laneIdx() == 0) partial_s[warpIdx()] = s;
    __syncthreads();
    if (warpIdx() == 0) {                                 // stage 2: warp 0
        unsigned int warps = blockDim.x / warpSize;
        s = laneIdx() < warps ? partial_s[laneIdx()] : 0.0f;
        s = warp_reduce(s);
    }

    float mean = s / cols;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = row[c] - mean;
}
