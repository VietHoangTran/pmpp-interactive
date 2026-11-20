// Lab 22.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Two ways to cut a decode-time matrix-vector product:
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major, batch <= MAX_BATCH.
// Same layout and argument order as the project's launchLinear (labs 6.4 and 15.6).
//
// Book: section 22.2 (pp. 532-533) names the two decompositions and the criteria for
// choosing between them; section 22.5 (p. 539) is why decode is a matrix-vector product.
// Chapter 22 prints no code (notes/ch22.md, "Code listings"), so both kernels are ours.
//   (a) gemvGatherKernel, output-centric (gather): one warp owns output column j and
//       accumulates in registers. No two threads write the same element: no atomics.
//   (b) gemvSplitKKernel, input-centric along `in`: `in` is cut into `splits` chunks and
//       one warp takes one (chunk, output) pair, so `splits` warps add into the same
//       Y[b][j]. That is a scatter, and the book's p. 533 says what a scatter needs:
//       atomic operations. In exchange it exposes `splits` times as many warps.
//
// Portability, not in the book:
//   - The lane mask has its own type. HIP's _sync intrinsics static_assert a 64-bit mask
//     on every target (notes/today/infra-facts.md B3); CUDA's is an unsigned int.
//   - The warp width is never a literal: warpSize in the kernels, the device property
//     in the launchers. It is 32 on NVIDIA and 64 or 32 on AMD (ch05 facts B5).
//   - On HIP, float atomicAdd compiles to a safe operation, possibly a compare-and-swap
//     loop, unless the build passes -munsafe-fp-atomics (infra-facts B1). The course's
//     HIP build does not.
#include <cstdio>

#define MAX_BATCH 8          // decode-sized; llama.cpp's mmvf kernels stop at 8 too (ch22-facts D6)
#define WARPS_PER_BLOCK 4    // 128 threads on a 32-lane warp, 256 on a 64-lane one

#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// Chapter 10's warp reduce: every lane calls it, lane 0 gets the warp's sum. Lanes whose
// partner lies past the end of the warp get their own value back (ch10 facts A3); only
// lane 0's result is used.
__device__ float warpSum(float v) {
    const lane_mask_t full = ~(lane_mask_t)0;
    for (unsigned int offset = warpSize / 2; offset > 0; offset /= 2) {
        v += __shfl_down_sync(full, v, offset);
    }
    return v;
}

// The warp width of device 0, read once.
static int deviceWarpSize() {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    return width;
}

// ---------------------------------------------------------------- one warp, one row of W
// The work both kernels share: one warp's dot products of a stretch [k0, k1) of row w of
// W with every batch row of X. Every lane of the warp calls it with the same arguments
// (and its own lane). On return lane 0's total[b] holds, for every b < batch,
//     the sum over k in [k0, k1) of X[b][k] * w[k];
// the other lanes' totals are not used.
__device__ void warpRowDots(const float* X, const float* w, int batch, int in, int k0, int k1, int lane,
                            float total[MAX_BATCH]) {
    // One running sum per batch row, in registers. MAX_BATCH is a constant and every
    // index below is too once the loops are unrolled, so the array need not live in
    // memory; `b < batch` switches the unused ones off.
    float sum[MAX_BATCH];
    #pragma unroll
    for (int b = 0; b < MAX_BATCH; ++b) {
        sum[b] = 0.0f;
    }

    // The lanes stride along [k0, k1) together: each step reads warpSize consecutive
    // floats of w, and of each row of X. Each W element is read once and used for every
    // batch row.
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
        // `b < batch` is the same for every lane, so every lane reaches the shuffles.
        total[b] = (b < batch) ? warpSum(sum[b]) : 0.0f;
    }
}

// ---------------------------------------------------------------- (a) gather
__global__ void gemvGatherKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    const int lane = threadIdx.x % warpSize;
    const int j = blockIdx.x * (blockDim.x / warpSize) + threadIdx.x / warpSize;   // this warp's output
    if (j >= out) {
        return;   // j is the same for every lane, so the whole warp leaves together
    }
    float total[MAX_BATCH];
    warpRowDots(X, W + (size_t)j * in, batch, in, 0, in, lane, total);   // the whole of row j
    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                Y[(size_t)b * out + j] = total[b];   // nobody else writes this element
            }
        }
    }
}

void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0) {
        return;
    }
    if (batch > MAX_BATCH) {
        std::fprintf(stderr, "launchGemvGather: batch %d is above MAX_BATCH (%d)\n", batch, MAX_BATCH);
        return;
    }
    const int width = deviceWarpSize();
    if (width <= 0) {
        std::fprintf(stderr, "launchGemvGather: could not read the device's warp width\n");
        return;
    }
    // One warp per output: ceil(out / WARPS_PER_BLOCK) blocks of WARPS_PER_BLOCK warps.
    const int blocks = (out + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
    gemvGatherKernel<<<blocks, WARPS_PER_BLOCK * width>>>(X, W, Y, batch, in, out);
}

// ---------------------------------------------------------------- (b) split-K
// Warp g of the grid takes output column j = g / splits and chunk s = g % splits of `in`.
// The `splits` warps that add into one output are therefore consecutive: with
// splits <= WARPS_PER_BLOCK they share a block, and with more they fill whole blocks
// whenever splits is a multiple of WARPS_PER_BLOCK. (The race emulator can only see two
// warps collide when they run at the same time, and it runs the warps of one block at the
// same time: README, "What each backend can catch".)
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
    float total[MAX_BATCH];
    warpRowDots(X, W + (size_t)j * in, batch, in, k0, k1, lane, total);   // chunk s of row j only
    if (lane == 0) {
        #pragma unroll
        for (int b = 0; b < MAX_BATCH; ++b) {
            if (b < batch) {
                // `splits` warps add into this element, so the read-modify-write must be
                // one indivisible operation. Relaxed and device-wide (infra-facts A1, B1).
                atomicAdd(&Y[(size_t)b * out + j], total[b]);
            }
        }
    }
}

// Provided. Y is zeroed first because the kernel adds into it.
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
