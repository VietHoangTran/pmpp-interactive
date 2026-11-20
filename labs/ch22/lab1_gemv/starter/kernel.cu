// Lab 22.1 starter (CUDA; also built for HIP and for the CPU emulators).
// Two ways to cut a decode-time matrix-vector product:
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major, batch <= MAX_BATCH.
// Same layout and argument order as the project's launchLinear (labs 6.4 and 15.6).
//
// Book: section 22.2 (pp. 532-533) names the two decompositions and the criteria for
// choosing between them; section 22.5 (p. 539) is why decode is a matrix-vector product.
// Chapter 22 prints no code, so both kernels are ours.
//   (a) gemvGatherKernel, output-centric (gather): yours to write.
//   (b) gemvSplitKKernel, input-centric along `in` (scatter): provided, with one defect.
//
// Portability, not in the book: the lane mask has its own type (HIP's _sync intrinsics
// insist on 64 bits, infra-facts B3), and the warp width is never a literal: warpSize in
// kernels, the device property in launchers (32 on NVIDIA, 64 or 32 on AMD, ch05 facts B5).
#include <cstdio>

#define MAX_BATCH 8          // decode-sized; llama.cpp's mmvf kernels stop at 8 too (ch22-facts D6)
#define WARPS_PER_BLOCK 4    // 128 threads on a 32-lane warp, 256 on a 64-lane one

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Chapter 10's warp reduce, provided: every lane calls it, lane 0 gets the warp's sum.
__device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(full, v, offset);
    }
    return v;
}

// The warp width of device 0, read once. Provided.
static int deviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

// ---------------------------------------------------------------- (a) gather: yours
__global__ void gemvGatherKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    // TODO 1: one warp owns one output column j, for every batch row.
    //   a. Find this thread's lane and its warp's output j from threadIdx.x, blockIdx.x,
    //      blockDim.x and warpSize. A warp whose j is past the end has nothing to do;
    //      can it return without leaving lanes of a shuffle behind?
    //   b. Keep one running sum per batch row in registers: an array of MAX_BATCH floats,
    //      with every loop over it bounded by the constant MAX_BATCH and `b < batch`
    //      tested inside, so that the compiler can keep it out of memory.
    //   c. The lanes stride along `in` together. Which order of reads lets each step of
    //      the warp read consecutive floats of row j of W? Read each W element once and
    //      use it for every batch row.
    //   d. Reduce each sum across the warp with warpSum. Which lanes must call it?
    //   e. Lane 0 writes Y[b][j]. No atomics: nobody else writes that element.
}

void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out) {
    // TODO 2: one warp per output. Blocks of WARPS_PER_BLOCK warps of deviceWarpSize()
    //         threads; how many blocks cover `out` outputs? Return without launching when
    //         there is nothing to compute, and refuse a batch above MAX_BATCH.
}

// ---------------------------------------------------------------- (b) split-K: provided
// This kernel has ONE defect. The emulator `emu` passes it; `race` does not. Find it with
// `race` (README, "What each backend can catch"), say why `emu` cannot see it, and fix it.
//
// Warp g of the grid takes output column j = g / splits and chunk s = g % splits of `in`.
// The `splits` warps that add into one output are therefore consecutive: with
// splits <= WARPS_PER_BLOCK they share a block, and with more they fill whole blocks
// whenever splits is a multiple of WARPS_PER_BLOCK.
__global__ void gemvSplitKKernel(const float* X, const float* W, float* Y, int batch, int in, int out,
                                 int splits) {
    const int lane = threadIdx.x % warpSize;
    const int g = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;
    if (g >= splits * out) {
        return;
    }
    const int j = g / splits;                          // the output this warp adds into
    const int s = g % splits;                          // the chunk of `in` it reads
    const int chunk = (in + splits - 1) / splits;      // ceil(in / splits)
    const int k0 = s * chunk;
    const int k1 = (k0 + chunk < in) ? k0 + chunk : in;
    if (k0 >= in) {
        return;   // an empty chunk (splits does not divide `in`, or splits > in): nothing to add
    }
    const float* w = W + (size_t)j * in;

    float sum[MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }
    for (int k = k0 + lane; k < k1; k += warpSize) {
        const float wk = w[k];
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                sum[b] += X[(size_t)b * in + k] * wk;
            }
        }
    }

    #pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
        if (b < batch) {
            const float total = warpSum(sum[b]);
            if (lane == 0) {
                Y[(size_t)b * out + j] += total;
            }
        }
    }
}

// Provided, and correct. Y is zeroed first because the kernel adds into it.
void launchGemvSplitK(const float* X, const float* W, float* Y, int batch, int in, int out, int splits) {
    if (batch <= 0 || out <= 0) {
        return;
    }
    if (batch > MAX_BATCH || splits < 1) {
        std::fprintf(stderr, "launchGemvSplitK: need 1 <= batch <= %d and splits >= 1 (got %d, %d)\n", MAX_BATCH,
                     batch, splits);
        return;
    }
    const int width = deviceWarpSize();
    if (width <= 0) {
        std::fprintf(stderr, "launchGemvSplitK: could not read the device's warp width\n");
        return;
    }
    cudaMemset(Y, 0, (size_t)batch * out * sizeof(float));   // ordered before the kernel in the same stream
    const long long warps = (long long)splits * out;
    const int blocks = (int)((warps + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK);
    gemvSplitKKernel<<<blocks, WARPS_PER_BLOCK * width>>>(X, W, Y, batch, in, out, splits);
}
