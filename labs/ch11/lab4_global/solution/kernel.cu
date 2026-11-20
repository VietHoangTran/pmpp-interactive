// Lab 11.4 reference solution (CUDA; also built for HIP and for the CPU emulators).
// A scan of any length, hierarchically. Book: section 11.9, pp. 273-276.
//
//   scanSegment            the register-tiled block scan of lab 11.2 (Fig. 11.13, p. 271),
//                          with a carry added to the segment's first element
//   ScanSegmentsKernel     Fig. 11.14 (p. 274), kernel 1: scan each segment, write its total
//   AddBlockSumsKernel     Fig. 11.14, kernel 3: add the total of all earlier segments
//   ReduceSegmentsKernel   Fig. 11.15 (p. 275), kernel 1: each segment's total, by a
//                          chapter 10 reduction (Figs. 10.16 and 10.18)
//   ScanWithCarryKernel    Fig. 11.15, kernel 3: add the earlier segments' total to the
//                          first element, then scan the segment
//   scratchFloats, launchScanScanAdd, launchReduceScanScan
//
// Contract: an inclusive scan with + of all n elements, 1 <= n < 2^31. One block of
// `block` threads owns a segment of S = COARSE_FACTOR*block consecutive elements. The
// block size is a power of two from 64 to 1024. An element at or past n loads 0, the
// identity of +, and is not stored. `scratch` holds scratchFloats(n, block) floats that
// the launchers may use as they like; nothing is assumed about what it holds on entry.
//
// Where this file differs from the book, and why:
//   - Figs. 11.14 and 11.15 are diagrams. The book's kernel 2 scans the block totals "in
//     a single block" (p. 273), which caps n at S*S elements. Here kernel 2 is the same
//     design applied to the totals, recursively, until they fit in one segment, so any n
//     works. The recursion adds a few more launches on arrays S times smaller.
//   - Fig. 11.13, the block scan underneath, never uses N (notes/ch11.md, "Unclear in the
//     PDF"). Every global load and store here is checked against N.
//   - p. 276 warns that some chapter 10 reductions assume a commutative operator. This one
//     adds in a different order from the scan, which is fine for + (floating-point + is
//     commutative; it is not associative, so the last bits can differ from Fig. 11.1's
//     loop either way).

#define COARSE_FACTOR 4      // elements per thread; the segment is COARSE_FACTOR*blockDim.x
#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents (ch11-facts B1)

// The driver asks for the factor, so it can size the scratch and print the book's byte
// counts. Change COARSE_FACTOR above if you like; keep COARSE_FACTOR*1024*4 bytes plus
// 4 KiB within the 48 KiB a block may use without opting in (ch05-facts A3).
int coarseFactor() { return COARSE_FACTOR; }

// ---------------------------------------------------------------- from labs 11.1 and 10.1
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
// thread. input and output may be the same array: each thread stores only the elements
// it loaded itself. Every thread of the block must call it.
__device__ float scanSegment(const float* input, float* output, float carry, unsigned int N,
                             float* buffer_s, float* threadSums, float* warpSums_s) {
    unsigned int blockSegment = blockIdx.x * COARSE_FACTOR * blockDim.x;

    // Coalesced load, chunk by chunk (p. 269). Out of range loads 0.
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        buffer_s[c * blockDim.x + threadIdx.x] = idx < N ? input[idx] : 0.0f;
    }
    // Fig. 11.15, kernel 3: the earlier segments' total goes into the first element, so
    // the scan carries it to every element. Thread 0 loaded element 0 itself.
    if (threadIdx.x == 0) {
        buffer_s[0] += carry;
    }
    __syncthreads();

    // Register-tiled run scan, block scan of the run totals, add (Fig. 11.13).
    unsigned int threadSegment = threadIdx.x * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
    buffer_r[0] = buffer_s[threadSegment];
#pragma unroll
    for (unsigned int c = 1; c < COARSE_FACTOR; ++c) {
        buffer_r[c] = buffer_s[threadSegment + c] + buffer_r[c - 1];
    }
    float threadSum = blockScan(buffer_r[COARSE_FACTOR - 1], warpSums_s);
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();

    float prevPartialSum = threadIdx.x > 0 ? threadSums[threadIdx.x - 1] : 0.0f;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        buffer_s[threadSegment + c] = buffer_r[c] + prevPartialSum;
    }
    __syncthreads();

    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx] = buffer_s[c * blockDim.x + threadIdx.x];
        }
    }
    return threadSums[blockDim.x - 1];   // the scan of all run totals: the segment's total
}

// ---------------------------------------------------------------- Fig. 11.14
// Kernel 1: scan each segment; write its total to blockSums[blockIdx.x].
__global__ void ScanSegmentsKernel(const float* input, float* output, float* blockSums,
                                   unsigned int N) {
    extern __shared__ float buffer_s[];                // COARSE_FACTOR*blockDim.x floats
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    float total = scanSegment(input, output, 0.0f, N, buffer_s, threadSums, warpSums_s);
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = total;
    }
}

// Kernel 3: add the scanned total of all earlier segments to every element of this one.
// scannedBlockSums[b] is the sum of segments 0..b, so segment b needs entry b - 1.
__global__ void AddBlockSumsKernel(float* output, const float* scannedBlockSums, unsigned int N) {
    if (blockIdx.x == 0) {
        return;                                        // nothing comes before segment 0
    }
    float carry = scannedBlockSums[blockIdx.x - 1];
    unsigned int blockSegment = blockIdx.x * COARSE_FACTOR * blockDim.x;
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            output[idx] += carry;
        }
    }
}

// ---------------------------------------------------------------- Fig. 11.15
// Kernel 1: each segment's total, without writing the segment. Each thread adds the
// elements it would have loaded for the scan (coalesced), then the two-stage warp-level
// reduction of lab 10.1.
__global__ void ReduceSegmentsKernel(const float* input, float* blockSums, unsigned int N) {
    __shared__ float partialSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    unsigned int blockSegment = blockIdx.x * COARSE_FACTOR * blockDim.x;
    float partialSum = 0.0f;
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = blockSegment + c * blockDim.x + threadIdx.x;
        if (idx < N) {
            partialSum += input[idx];
        }
    }
    partialSum = warpReduce(partialSum);
    if (laneIdx() == 0) {
        partialSums_s[warpIdx()] = partialSum;
    }
    __syncthreads();
    if (warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        partialSum = laneIdx() < warps ? partialSums_s[laneIdx()] : 0.0f;
        partialSum = warpReduce(partialSum);
        if (threadIdx.x == 0) {
            blockSums[blockIdx.x] = partialSum;
        }
    }
}

// Kernel 3: the earlier segments' total into the first element, then scan the segment.
// The output is final; nothing is written back to scratch.
__global__ void ScanWithCarryKernel(const float* input, float* output,
                                    const float* scannedBlockSums, unsigned int N) {
    extern __shared__ float buffer_s[];
    __shared__ float threadSums[MAX_BLOCK_DIM];
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE];
    float carry = (threadIdx.x == 0 && blockIdx.x > 0) ? scannedBlockSums[blockIdx.x - 1] : 0.0f;
    scanSegment(input, output, carry, N, buffer_s, threadSums, warpSums_s);
}

// ---------------------------------------------------------------- host side
static unsigned int segmentsOf(unsigned int n, unsigned int block) {
    unsigned int segment = COARSE_FACTOR * block;
    return (n + segment - 1) / segment;
}

// One float per segment, at every level of the recursion: the n elements have
// segmentsOf(n) totals, those totals have segmentsOf(segmentsOf(n)) totals, and so on
// down to the level with one segment, whose total scan-scan-add also writes.
unsigned int scratchFloats(unsigned int n, unsigned int block) {
    unsigned int total = 0;
    do {
        n = segmentsOf(n, block);
        total += n;
    } while (n > 1);
    return total;
}

// Fig. 11.14. Level by level: scratch[0 .. blocks) holds this level's totals, the rest
// of scratch belongs to the levels below.
void launchScanScanAdd(const float* in, float* out, float* scratch, unsigned int n,
                       unsigned int block) {
    unsigned int blocks = segmentsOf(n, block);
    size_t shared = COARSE_FACTOR * block * sizeof(float);
    float* blockSums = scratch;
    ScanSegmentsKernel<<<blocks, block, shared>>>(in, out, blockSums, n);         // kernel 1
    if (blocks > 1) {
        launchScanScanAdd(blockSums, blockSums, scratch + blocks, blocks, block);  // kernel 2
        AddBlockSumsKernel<<<blocks, block>>>(out, blockSums, n);                  // kernel 3
    }
}

// Fig. 11.15, with the same scratch layout. With a single segment there is nothing to
// carry: kernel 3 alone scans it (and never reads scratch).
void launchReduceScanScan(const float* in, float* out, float* scratch, unsigned int n,
                          unsigned int block) {
    unsigned int blocks = segmentsOf(n, block);
    size_t shared = COARSE_FACTOR * block * sizeof(float);
    float* blockSums = scratch;
    if (blocks > 1) {
        ReduceSegmentsKernel<<<blocks, block>>>(in, blockSums, n);                    // kernel 1
        launchReduceScanScan(blockSums, blockSums, scratch + blocks, blocks, block);  // kernel 2
    }
    ScanWithCarryKernel<<<blocks, block, shared>>>(in, out, blockSums, n);           // kernel 3
}
