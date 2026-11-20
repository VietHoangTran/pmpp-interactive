// Provided for lab 11.2, not edited: the coarsened, register-tiled scan with NO staging
// through shared memory. Each thread loads its own run of COARSE_FACTOR consecutive
// elements straight from global memory, scans it in registers, and stores it straight
// back. In one load instruction the lanes of a warp touch addresses COARSE_FACTOR floats
// apart: the pattern p. 269 warns is uncoalesced, and the reason Figs. 11.12 and 11.13
// go through shared memory. llama.cpp's CUDA cumsum loads this way (notes/today/
// ch11-facts.md D1), so it is a real design, not a straw man. Lab 11.2 times it against
// your staged kernels.
//
// Not a figure of the book: it is Fig. 11.13 with lines 05-09 and 31-36 replaced by
// direct loads and stores. Same contract as lab 11.2's kernels: block b scans its own
// segment of COARSE_FACTOR*blockDim.x elements; elements at or past N load 0 and are
// not stored.
//
// The block scan below is a private copy of lab 11.1's (Figs. 11.8 and 11.9, made
// portable), with `static` names so that it cannot clash with yours at link time.

#define TM_MAX_BLOCK_DIM 1024
#define TM_MIN_WARP_SIZE 32     // the narrowest warp any backend here documents

#if defined(__HIPCC__)
typedef unsigned long long tm_lane_mask_t;   // HIP: 64-bit masks only (infra-facts B3)
#else
typedef unsigned int tm_lane_mask_t;
#endif

static __device__ __forceinline__ unsigned int tm_laneIdx() { return threadIdx.x % warpSize; }
static __device__ __forceinline__ unsigned int tm_warpIdx() { return threadIdx.x / warpSize; }

static __device__ __forceinline__ float tm_warpScan(float val) {
    const tm_lane_mask_t full = ~(tm_lane_mask_t)0;
    for (unsigned int stride = 1; stride < (unsigned int)warpSize; stride *= 2) {
        float leftVal = __shfl_up_sync(full, val, stride);
        if (tm_laneIdx() >= stride) {
            val += leftVal;
        }
    }
    return val;
}

static __device__ __forceinline__ float tm_blockScan(float val, float* warpSums_s) {
    val = tm_warpScan(val);
    if (tm_laneIdx() == (unsigned int)warpSize - 1) {
        warpSums_s[tm_warpIdx()] = val;
    }
    __syncthreads();
    if (tm_warpIdx() == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float warpSum = tm_laneIdx() < warps ? warpSums_s[tm_laneIdx()] : 0.0f;
        warpSum = tm_warpScan(warpSum);
        if (tm_laneIdx() < warps) {
            warpSums_s[tm_laneIdx()] = warpSum;
        }
    }
    __syncthreads();
    if (tm_warpIdx() > 0) {
        val += warpSums_s[tm_warpIdx() - 1];
    }
    return val;
}

template <unsigned int COARSE_FACTOR>
__global__ void ThreadMajorScanKernel(const float* input, float* output, unsigned int N) {
    __shared__ float threadSums[TM_MAX_BLOCK_DIM];
    __shared__ float warpSums_s[TM_MAX_BLOCK_DIM / TM_MIN_WARP_SIZE];

    // This thread's run: COARSE_FACTOR consecutive elements, read directly.
    unsigned int threadSegment = blockIdx.x * COARSE_FACTOR * blockDim.x + threadIdx.x * COARSE_FACTOR;
    float buffer_r[COARSE_FACTOR];
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = threadSegment + c;
        float x = idx < N ? input[idx] : 0.0f;
        buffer_r[c] = c == 0 ? x : buffer_r[c - 1] + x;
    }

    float threadSum = tm_blockScan(buffer_r[COARSE_FACTOR - 1], warpSums_s);
    threadSums[threadIdx.x] = threadSum;
    __syncthreads();

    float prevPartialSum = threadIdx.x > 0 ? threadSums[threadIdx.x - 1] : 0.0f;
#pragma unroll
    for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
        unsigned int idx = threadSegment + c;
        if (idx < N) {
            output[idx] = buffer_r[c] + prevPartialSum;
        }
    }
}

void launchThreadMajorScan(const float* in, float* out, int n, int block, int coarse) {
    unsigned int segment = (unsigned int)coarse * block;
    unsigned int blocks = ((unsigned int)n + segment - 1) / segment;
    switch (coarse) {
        case 1:  ThreadMajorScanKernel<1><<<blocks, block>>>(in, out, (unsigned int)n); break;
        case 2:  ThreadMajorScanKernel<2><<<blocks, block>>>(in, out, (unsigned int)n); break;
        case 4:  ThreadMajorScanKernel<4><<<blocks, block>>>(in, out, (unsigned int)n); break;
        case 8:  ThreadMajorScanKernel<8><<<blocks, block>>>(in, out, (unsigned int)n); break;
        case 16: ThreadMajorScanKernel<16><<<blocks, block>>>(in, out, (unsigned int)n); break;
        default: break;   // the driver uses only these five factors
    }
}
