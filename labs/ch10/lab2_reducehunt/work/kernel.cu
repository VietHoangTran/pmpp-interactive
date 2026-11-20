// Lab 10.2 starter: a sum over any length n (after book Figs. 10.13, 10.16 and 10.18,
// pp. 240, 243 and 245). Blocks of BLOCK_DIM threads; each thread adds two elements,
// each warp reduces its values with shuffles, warp 0 reduces the per-warp sums, and
// thread 0 of every block adds the block's total into *out, which the launcher zeroes.
//
// This kernel has FOUR independent defects. It compiles, and it can pass for some
// lengths. Find and fix all four.
// Run it on the CPU emulator first (--backend emu): its thread orders are fixed, so a
// synchronisation defect fails every time there instead of occasionally. Then run the
// race backend, then your GPU. One of the four is invisible to emu.
//
// Where this file differs from the page, on purpose and not a defect: the lane mask has
// its own type and the shuffle loop starts at warpSize/2, because HIP rejects the
// book's 32-bit 0xffffffff and AMD warps may have 64 lanes (notes/today/ch10-facts.md B1).

#define BLOCK_DIM 128        // threads per block; each block reduces 2*BLOCK_DIM elements
#define MIN_WARP_SIZE 32     // the narrowest warp a CUDA or HIP device has (ch10-facts B2)

// Fig. 10.11's first parameter. CUDA: unsigned. HIP: every _sync intrinsic
// static_asserts a 64-bit mask, in wave32 mode too (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// Fig. 10.13. Every lane of the warp must call it; lane 0 returns the warp's sum.
__device__ __inline__ float warp_reduce(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;             // every lane of the warp
    float partialSum = val;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        partialSum += __shfl_down_sync(full, partialSum, stride);
    }
    return partialSum;
}

__global__ void reduceAnyKernel(const float* in, float* out, unsigned int n) {
    unsigned int segment = 2 * BLOCK_DIM * blockIdx.x;
    unsigned int i = segment + threadIdx.x;

    if (i >= n) return;                                   // nothing to add here
    float partialSum = in[i];
    if (i + BLOCK_DIM < n) partialSum += in[i + BLOCK_DIM];

    // Stage 1: every warp reduces its own values in registers.
    partialSum = warp_reduce(partialSum);

    // One slot per warp, sized for the narrowest warp.
    __shared__ float partials_s[BLOCK_DIM / MIN_WARP_SIZE];
    if (laneIdx() == 0) {
        partials_s[warpIdx()] = partialSum;
    }

    // Stage 2: warp 0 reduces the per-warp sums.
    if (warpIdx() == 0) {
        float blockSum = partials_s[laneIdx()];
        blockSum = warp_reduce(blockSum);
        if (threadIdx.x == 0) {
            *out += blockSum;                             // commit the block's total
        }
    }
}

// *out = the sum of in[0 .. n-1].
void launchReduceAny(const float* in, float* out, int n) {
    cudaMemset(out, 0, sizeof(float));           // 0.0f, the identity: every block adds to it
    if (n <= 0) return;
    unsigned int blocks = ((unsigned int)n + 2 * BLOCK_DIM - 1) / (2 * BLOCK_DIM);
    reduceAnyKernel<<<blocks, BLOCK_DIM>>>(in, out, (unsigned int)n);
}
