// Lab 11.2 starter (CUDA; also built for HIP and for the CPU emulators).
// Coarsening and register tiling. Book: sections 11.6 and 11.7, pp. 267-272.
//
//   (a) CoarsenedScanKernel       Fig. 11.12, p. 268
//   (b) RegisterTiledScanKernel   Fig. 11.13, p. 271
//       blockScan                 Fig. 11.9, p. 264, as you wrote it in lab 11.1
//
// Contract: block b scans its own segment of COARSE_FACTOR*blockDim.x elements,
// starting at b*COARSE_FACTOR*blockDim.x, inclusive, with +, independently of every other
// block. Any n >= 1. An element at or past n loads 0, the identity of +, and is not
// stored. Block sizes are powers of two from 64 to 1024; COARSE_FACTOR is 1, 2, 4, 8 or
// 16, with COARSE_FACTOR*blockDim.x at most 8192 (the driver asks for nothing larger).
//
// Where the book's listings will not do as printed (the lesson pages say why):
//   - Figs. 11.12 and 11.13 take N but never use it.
//   - Their BLOCK_DIM is a compile-time constant; here the block size is blockDim.x,
//     chosen at run time.
//   - COARSE_FACTOR is a macro on the page. Here it is a template parameter, so one
//     build can test and time every factor; the launchers pick the instantiation.
//
// Shared memory: the segment buffer (COARSE_FACTOR*blockDim.x floats) is dynamic shared
// memory, `extern __shared__ float buffer_s[];`, sized by the third <<< >>> argument.
// A static array would have to be sized for the largest block, and a 256-thread block
// would then reserve four times what it uses. threadSums and the warp totals are static,
// sized for the largest block. With COARSE_FACTOR*blockDim.x <= 8192 the total stays
// within the 48 KiB a block may use without opting in.

#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents

// ---------------------------------------------------------------- from lab 11.1
// TODO 0: replace this stub with lane_mask_t, laneIdx(), warpIdx(), warpScan and
//         blockScan from your lab 11.1 kernel.cu, unchanged.
__device__ __forceinline__ float blockScan(float val, float* warpSums_s) {
    return val;
}

// ---------------------------------------------------------------- (a)
template <unsigned int COARSE_FACTOR>
__global__ void CoarsenedScanKernel(const float* input, float* output, unsigned int N) {
    // TODO 1: Fig. 11.12, with every global access checked against N.
    //   a. declare the dynamic segment buffer, threadSums, and the warp totals blockScan
    //      needs
    //   b. load the segment in COARSE_FACTOR passes of blockDim.x elements. In pass c,
    //      which element does thread t load, so that the threads of a warp read
    //      consecutive addresses? Where does it go in buffer_s? Past N, load 0.
    //   c. a barrier: why can no thread start step d before every thread has finished b?
    //   d. thread t owns the run of COARSE_FACTOR elements starting at t*COARSE_FACTOR.
    //      Scan it sequentially, in place in buffer_s.
    //   e. blockScan the run totals (every thread calls it), keep each thread's scanned
    //      total in threadSums, barrier
    //   f. add the scanned total of the runs before yours to every element of your run
    //   g. a barrier, then store in passes, as you loaded. Nothing at or past N.
}

void launchCoarsenedScan(const float* in, float* out, int n, int block, int coarse) {
    // TODO 2: one block of `block` threads per segment of coarse*block elements, enough
    //         blocks to cover all n, and the bytes of the dynamic segment buffer as the
    //         third <<< >>> argument. COARSE_FACTOR is a template argument, so choose the
    //         instantiation with a switch over the five factors: 1, 2, 4, 8 and 16.
}

// ---------------------------------------------------------------- (b)
template <unsigned int COARSE_FACTOR>
__global__ void RegisterTiledScanKernel(const float* input, float* output, unsigned int N) {
    // TODO 3: Fig. 11.13. The same load, block scan and store as (a). What changes is
    //         the work on the thread's run:
    //   a. copy the run into a local array buffer_r[COARSE_FACTOR], scanning as you go,
    //      so each element is read from shared memory once
    //   b. after the block scan, add the earlier runs' total to buffer_r and write the
    //      result back to buffer_s in the same pass. Thread 0 adds 0.
    //   Put #pragma unroll before the loops over buffer_r. Why does it matter whether
    //   every index into buffer_r is a compile-time constant (p. 271)?
}

void launchRegisterTiledScan(const float* in, float* out, int n, int block, int coarse) {
    // TODO 4: as TODO 2.
}
