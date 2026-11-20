// Lab 11.4 starter (CUDA; also built for HIP and for the CPU emulators).
// A scan of any length, hierarchically. Book: section 11.9, pp. 273-276.
//
//   scanSegment            the register-tiled block scan of lab 11.2 (Fig. 11.13, p. 271),
//                          with a carry added to the segment's first element
//   ScanSegmentsKernel     Fig. 11.14 (p. 274), kernel 1
//   AddBlockSumsKernel     Fig. 11.14, kernel 3
//   ReduceSegmentsKernel   Fig. 11.15 (p. 275), kernel 1, a chapter 10 reduction
//   ScanWithCarryKernel    Fig. 11.15, kernel 3
//   scratchFloats, launchScanScanAdd, launchReduceScanScan
//
// Contract: an inclusive scan with + of all n elements, 1 <= n < 2^31. One block of
// `block` threads owns a segment of S = COARSE_FACTOR*block consecutive elements: block b
// owns elements b*S to b*S + S - 1. The block size is a power of two from 64 to 1024. An
// element at or past n loads 0, the identity of +, and is not stored. `scratch` holds
// scratchFloats(n, block) floats that the launchers may use as they like; nothing is
// assumed about what it holds on entry.
//
// The book draws Figs. 11.14 and 11.15 but prints no code for them, and its kernel 2
// scans the block totals "in a single block" (p. 273). That caps n at S*S. Here kernel 2
// is the same three-kernel design applied to the totals, again and again until they fit
// in one segment, so any n works.

#define COARSE_FACTOR 4      // elements per thread; the segment is COARSE_FACTOR*blockDim.x
#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (ch11-facts B1)

// The driver asks for the factor, so it can size buffers and print the book's byte
// counts. Change COARSE_FACTOR above if you like; keep COARSE_FACTOR*1024*4 bytes plus
// 4 KiB within the 48 KiB a block may use without opting in (ch05-facts A3).
int coarseFactor() { return COARSE_FACTOR; }

// ---------------------------------------------------------------- from labs 11.1 and 10.1
// Provided: the lab 11.1 block scan and the lab 10.1 warp reduction. Use yours instead if
// you like; keep the signatures.
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;   // HIP: 64-bit masks only (infra-facts B3)
#else
typedef unsigned int lane_mask_t;
#endif

__device__ __forceinline__ unsigned int laneIdx() { return threadIdx.x % warpSize; }
__device__ __forceinline__ unsigned int warpIdx() { return threadIdx.x / warpSize; }

__device__ __forceinline__ float warpScan(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = 1; stride < (unsigned int)warpSize; stride *= 2) {
        float leftVal = __shfl_up_sync(full, val, stride);
        if (laneIdx() >= stride) {
            val += leftVal;
        }
    }
    return val;
}

__device__ __forceinline__ float blockScan(float val, float* warpSums_s) {
    val = warpScan(val);
    if (laneIdx() == (unsigned int)warpSize - 1) {
        warpSums_s[warpIdx()] = val;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float warpSum = laneIdx() < warps ? warpSums_s[laneIdx()] : 0.0f;
        warpSum = warpScan(warpSum);
        if (laneIdx() < warps) {
            warpSums_s[laneIdx()] = warpSum;
        }
    }
    __syncthreads();
    if (warpIdx() > 0) {
        val += warpSums_s[warpIdx() - 1];
    }
    return val;
}

// Lane 0 returns the sum of the warp's values (lab 10.1, Fig. 10.13).
__device__ __forceinline__ float warpReduce(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int stride = warpSize / 2; stride > 0; stride /= 2) {
        val += __shfl_down_sync(full, val, stride);
    }
    return val;
}

// ---------------------------------------------------------------- the block-level scan
// Scans this block's segment of `input` into `output`, after adding `carry` to the
// segment's first element, and returns the segment's total (carry included) in every
// thread. Every thread of the block must call it. The shared arrays are declared by the
// kernels and passed in: buffer_s holds COARSE_FACTOR*blockDim.x floats, threadSums
// MAX_BLOCK_DIM, warpSums_s MAX_BLOCK_DIM/MIN_WARP_SIZE.
__device__ float scanSegment(const float* input, float* output, float carry, unsigned int N,
                             float* buffer_s, float* threadSums, float* warpSums_s) {
    // TODO 1: start from your RegisterTiledScanKernel of lab 11.2: coalesced load (an
    //         element at or past N loads 0), register-tiled run scan, blockScan of the
    //         run totals, add, coalesced store (nothing at or past N). Then:
    //   a. Fig. 11.15 adds the earlier segments' total to the segment's FIRST element
    //      and lets the scan carry it to the others. Where in the code, and by which
    //      thread, can that be done without one more barrier?
    //   b. Which value, readable by every thread once the scan is done, is the segment's
    //      total? Return it.
    //   The recursion will call this with input == output (scanning the totals in
    //   place). Is that safe? Look at which elements each thread loads and stores.
    return 0.0f;
}

// ---------------------------------------------------------------- Fig. 11.14
// Kernel 1: scan each segment; write its total to blockSums[blockIdx.x].
__global__ void ScanSegmentsKernel(const float* input, float* output, float* blockSums,
                                   unsigned int N) {
    extern __shared__ float buffer_s[];                // COARSE_FACTOR*blockDim.x floats
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    // TODO 2: scan the segment (no carry here) and have one thread write its total.
}

// Kernel 3: add the scanned total of all earlier segments to every element of this one.
// After kernel 2, scannedBlockSums[b] is the sum of segments 0 to b.
__global__ void AddBlockSumsKernel(float* output, const float* scannedBlockSums, unsigned int N) {
    // TODO 3: which entry does segment b need, and what does segment 0 do? Touch the
    //         elements in the same coalesced order as the scan, COARSE_FACTOR per
    //         thread, and nothing at or past N.
}

// ---------------------------------------------------------------- Fig. 11.15
// Kernel 1: each segment's total, without writing the segment.
__global__ void ReduceSegmentsKernel(const float* input, float* blockSums, unsigned int N) {
    __shared__ float partialSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    // TODO 4: a chapter 10 reduction over the same segment the scan uses: each thread
    //         adds the COARSE_FACTOR elements it would load for the scan (coalesced; 0
    //         past N), then the two-stage warp-level reduction of lab 10.1 with
    //         warpReduce. One thread writes blockSums[blockIdx.x].
}

// Kernel 3: the earlier segments' total into the first element, then scan the segment.
__global__ void ScanWithCarryKernel(const float* input, float* output,
                                    const float* scannedBlockSums, unsigned int N) {
    extern __shared__ float buffer_s[];
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    // TODO 5: which carry does block b need? Block 0 must not read scannedBlockSums.
}

// ---------------------------------------------------------------- host side
static unsigned int segmentsOf(unsigned int n, unsigned int block) {
    unsigned int segment = COARSE_FACTOR * block;
    return (n + segment - 1) / segment;
}

// How many floats of scratch the launchers below need for n elements.
unsigned int scratchFloats(unsigned int n, unsigned int block) {
    // TODO 6: one float per segment at every level of the recursion. n elements have
    //         segmentsOf(n, block) totals; scanning those totals makes totals of its
    //         own; and so on. Where does it stop, and does your launcher write a total
    //         at the last level? The driver puts guard words after this many floats and
    //         reports any your launchers write.
    return 0;
}

// Fig. 11.14. Kernel 1 scans the segments and writes their totals; kernel 2 scans the
// totals (this same function, on the totals, in place); kernel 3 adds.
void launchScanScanAdd(const float* in, float* out, float* scratch, unsigned int n,
                       unsigned int block) {
    // TODO 7: one block per segment, with COARSE_FACTOR*block floats of dynamic shared
    //         memory for the scan kernels (the third <<< >>> argument). Lay scratch out
    //         level by level: this level's totals first, the rest for the levels below.
    //         When is there nothing left to scan and add?
}

// Fig. 11.15. Kernel 1 reduces the segments; kernel 2 scans the totals (this same
// function, in place); kernel 3 scans the segments with their carries.
void launchReduceScanScan(const float* in, float* out, float* scratch, unsigned int n,
                          unsigned int block) {
    // TODO 8: the same scratch layout. With a single segment, which kernels are needed?
}
