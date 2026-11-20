// Lab 10.2 reference solution: the four defects of the starter, fixed.
// A sum over any length n: blocks of BLOCK_DIM threads, each block reducing a segment
// of 2*BLOCK_DIM elements with the book's two-stage warp reduction and committing its
// total with an atomic add (book Figs. 10.13, 10.16 and 10.18, pp. 240, 243 and 245,
// with exercise 5's bounds checks, p. 250).
//
// Where this file differs from the page, and why:
//   - Fig. 10.13 passes the literal 0xffffffff and starts its loop at WARP_SIZE/2 = 16.
//     HIP rejects a 32-bit mask at compile time and AMD warps may have 64 lanes
//     (notes/today/ch10-facts.md B1), so the mask has its own type and the loop starts
//     at warpSize/2.
//   - Fig. 10.16 lets every lane of warp 0 read partialSums_s[t] although only one entry
//     per warp was written (FIX 2 below; ch10-facts B2).
//   - Fig. 10.18 declares N and never uses it; here every load is checked (FIX 1).
//   - The book's partialSums_s is called partials_s here.
//   - On HIP the commit is the legacy atomicAdd, not cuda::atomic_ref (FIX 4).

// cuda::atomic_ref for the commit. On HIP, <cuda/atomic> comes from libhipcxx, which
// this course's HIP build does not link (infra-facts B2).
#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

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
// A lane whose partner lies past the end of the warp gets its own value back
// (ch10-facts A3), so the upper lanes compute sums nobody reads.
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

    // FIX 1: no thread leaves early. The starter returned here when i >= n, and the
    // shuffles below then read registers of lanes that had exited, which is undefined
    // (ch10-facts A3); a warp whose lanes had all exited also never wrote its slot.
    // A thread past the end contributes 0, the identity of +, and stays for the shuffles.
    float partialSum = 0.0f;
    if (i < n) partialSum += in[i];
    if (i + BLOCK_DIM < n) partialSum += in[i + BLOCK_DIM];

    // Stage 1: every warp reduces its own values in registers.
    partialSum = warp_reduce(partialSum);

    // One slot per warp. Sized for the narrowest warp, so it is big enough on every
    // backend; with 64-wide warps half of it is never written.
    __shared__ float partials_s[BLOCK_DIM / MIN_WARP_SIZE];
    if (laneIdx() == 0) {
        partials_s[warpIdx()] = partialSum;
    }
    // FIX 3: warp 0 reads slots that lane 0 of every other warp wrote. Without this
    // barrier nothing orders those writes before the reads.
    __syncthreads();

    // Stage 2: warp 0 reduces the per-warp sums.
    if (warpIdx() == 0) {
        // FIX 2: only the first `warps` slots were written. The starter read
        // partials_s[lane] in every lane, as Fig. 10.16 does: past the end of the array
        // whenever warp 0 has more lanes than the array has slots (every block under
        // 1024 threads; on 64-wide warps, every block). The other lanes contribute 0.
        unsigned int warps = (blockDim.x + warpSize - 1) / warpSize;
        float blockSum = (laneIdx() < warps) ? partials_s[laneIdx()] : 0.0f;
        blockSum = warp_reduce(blockSum);
        if (threadIdx.x == 0) {
            // FIX 4: thread 0 of every block adds into the same *out. The starter's
            // `*out += blockSum` is a read, an add and a write; two blocks doing it at
            // once can both read the old value, and one sum is lost. The book's atomic
            // commit (Fig. 10.18). HIP takes the legacy call, which is also relaxed and
            // device-wide (infra-facts A1, B1); it would do on CUDA too.
#if defined(__HIPCC__)
            atomicAdd(out, blockSum);
#else
            cuda::atomic_ref<float, cuda::thread_scope_device> out_ref(*out);
            out_ref.fetch_add(blockSum, cuda::memory_order_relaxed);
#endif
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
