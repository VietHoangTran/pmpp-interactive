// Fig. 10.13 (p. 240) as printed, with the two helpers of pp. 238-239. CUDA only: under
// HIP the 32-bit mask fails a static_assert that asks for a 64-bit one, and the loop
// assumes a warp of 32 lanes. Section 10.7 shows it beside the portable version.
#define WARP_SIZE 32  // the book's assumption: every warp has 32 lanes

__device__ unsigned int warpIdx() { return threadIdx.x/WARP_SIZE; }
__device__ unsigned int laneIdx() { return threadIdx.x%WARP_SIZE; }

__device__ __inline__ float warp_reduce(float val) {
    float partialSum = val;
    for(unsigned int stride = WARP_SIZE/2; stride > 0; stride /= 2) {
        partialSum += __shfl_down_sync(0xffffffff, partialSum, stride);
    }
    return partialSum;
}
