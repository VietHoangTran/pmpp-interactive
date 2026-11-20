// Lab 11.3 starter: a row-wise EXCLUSIVE scan, out[r][j] = in[r][0] + ... + in[r][j-1],
// with out[r][0] = 0, in the shape llama.cpp, MLX and PyTorch ship (ch11-facts D1, D3, D4).
// One block per row. The block walks its row in chunks of COARSE_FACTOR*blockDim.x
// elements and carries the sum of the chunks before. Inside a chunk it is the book's
// coarsened scan (Figs. 11.12 and 11.13, pp. 268 and 271): a coalesced load staged through
// shared memory, a sequential scan of each thread's run in registers, then a block scan
// of the run totals built from warp scans (Figs. 11.8 and 11.9, p. 264).
//
// This kernel has THREE independent defects. It compiles, and it gets some shapes right.
// Find and fix all three. Run the CPU emulator first (--backend emu): its thread orders are
// fixed, so most synchronisation defects fail there every time. Then the race backend,
// then your GPU. Note which defect each of them showed you (README.md has a table).
//
// Where this file differs from the page, on purpose and not a defect:
//   - Exclusive, not inclusive (the book converts between the two on p. 253).
//   - The lane mask has its own type and the warp loop runs to warpSize: HIP rejects the
//     book's 32-bit 0xffffffff and AMD warps may have 64 lanes (notes/today/ch11-facts.md B1).
//   - Every load and store is bounds-checked; Figs. 11.12 and 11.13 have no checks.
//   - Each warp stages its own slice of the chunk instead of the whole block segment.
//   - The previous lane's inclusive sum comes from a shuffle, not from a shared array.
//   - The shared arrays are declared in the kernel, not in a helper (ch11-facts C2).

#include <cstdio>

#define COARSE_FACTOR 4      // elements per thread per chunk (the book's example factor, p. 269)
#define MAX_BLOCK_DIM 1024   // the largest block the launcher accepts
#define MIN_WARP_SIZE 32     // the narrowest warp a CUDA or HIP device has (ch10-facts B2)
#define MAX_WARP_SIZE 64     // the widest (AMD gfx9, ch05-facts B5)

// Fig. 11.8's first parameter. CUDA: unsigned. HIP: every _sync intrinsic
// static_asserts a 64-bit mask, in wave32 mode too (infra-facts B3).
#if defined(__HIPCC__)
typedef unsigned long long lane_mask_t;
#else
typedef unsigned int lane_mask_t;
#endif

// The book's helpers (pp. 238-239), with warpSize in place of the constant 32.
__device__ unsigned int warpIdx() { return threadIdx.x / warpSize; }
__device__ unsigned int laneIdx() { return threadIdx.x % warpSize; }

// Fig. 11.8: inclusive Kogge-Stone scan across the lanes of one warp. Every lane of the
// warp must call it.
__device__ __inline__ float warpScan(float val) {
    const lane_mask_t full = ~(lane_mask_t)0;            // every lane of the warp
    for (unsigned int stride = 1; stride < warpSize; stride *= 2) {
        float leftVal = __shfl_up_sync(full, val, stride);
        if (threadIdx.x >= stride) {
            val += leftVal;
        }
    }
    return val;
}

__global__ void ExclusiveRowScanKernel(const float* in, float* out, unsigned int cols) {
    __shared__ float stage_s[COARSE_FACTOR * MAX_BLOCK_DIM];   // each warp's slice of the chunk
    __shared__ float warpSums_s[MAX_BLOCK_DIM / MIN_WARP_SIZE]; // one slot per warp, reused every chunk

    const float* rowIn = in + (size_t)blockIdx.x * cols;
    float* rowOut = out + (size_t)blockIdx.x * cols;
    const lane_mask_t full = ~(lane_mask_t)0;
    const unsigned int lane = laneIdx();
    const unsigned int warp = warpIdx();
    const unsigned int numWarps = blockDim.x / warpSize;
    const unsigned int chunkSize = COARSE_FACTOR * blockDim.x;
    const unsigned int warpSlice = warp * COARSE_FACTOR * warpSize;   // where this warp's slice starts
    float* myStage = stage_s + warpSlice;

    float carry = 0.0f;   // the sum of every earlier chunk of this row
    for (unsigned int chunk = 0; chunk < cols; chunk += chunkSize) {
        // 1. Coalesced load: in step c the warp reads warpSize consecutive elements.
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int j = chunk + warpSlice + c * warpSize + lane;
            myStage[c * warpSize + lane] = (j < cols) ? rowIn[j] : 0.0f;
        }

        // 2. Each lane owns COARSE_FACTOR consecutive elements: an exclusive sequential scan
        //    of its run in registers (the register tile of Fig. 11.13).
        float run[COARSE_FACTOR];
        float threadSum = 0.0f;
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            run[c] = threadSum;
            threadSum += myStage[lane * COARSE_FACTOR + c];
        }

        // 3. Scan the run totals across the warp; the last lane holds the warp's total.
        float inclusive = warpScan(threadSum);
        float exclusive = __shfl_up_sync(full, inclusive, 1);   // the lanes before mine
        if (lane == 0) {
            exclusive = 0.0f;
        }
        if (lane == warpSize - 1) {
            warpSums_s[warp] = inclusive;
        }
        __syncthreads();

        // 4. Warp 0 scans the warp totals (Fig. 11.9). It needs numWarps <= warpSize,
        //    which the launcher guarantees.
        if (warp == 0) {
            float warpSum = (lane < numWarps) ? warpSums_s[lane] : 0.0f;
            warpSum = warpScan(warpSum);
            if (lane < numWarps) {
                warpSums_s[lane] = warpSum;
            }
        }
        __syncthreads();

        // 5. Everything before my run: earlier chunks, earlier warps, earlier lanes.
        float before = carry + ((warp > 0) ? warpSums_s[warp - 1] : 0.0f) + exclusive;
        carry += warpSums_s[numWarps - 1];                       // this chunk's total
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            myStage[lane * COARSE_FACTOR + c] = before + run[c];
        }
        __syncwarp();

        // 6. Coalesced store, the load's layout in reverse.
        for (unsigned int c = 0; c < COARSE_FACTOR; ++c) {
            unsigned int j = chunk + warpSlice + c * warpSize + lane;
            if (j < cols) {
                rowOut[j] = myStage[c * warpSize + lane];
            }
        }
    }
}

// out[r][j] = in[r][0] + ... + in[r][j-1] for every row r; out[r][0] = 0.
// `block` threads per block, one block per row: a multiple of 64, so that blocks are
// whole warps on every CUDA and HIP device, and at most MAX_BLOCK_DIM.
// (The launcher is correct; the defects are all in the kernel.)
void launchExclusiveRowScan(const float* in, float* out, int rows, int cols, int block) {
    if (rows <= 0 || cols <= 0) return;
    if (block <= 0 || block > MAX_BLOCK_DIM || block % MAX_WARP_SIZE != 0) {
        std::fprintf(stderr, "launchExclusiveRowScan: %d threads per block is not a multiple of %d up to %d\n",
                     block, MAX_WARP_SIZE, MAX_BLOCK_DIM);
        return;
    }
    ExclusiveRowScanKernel<<<rows, block>>>(in, out, (unsigned int)cols);
}
