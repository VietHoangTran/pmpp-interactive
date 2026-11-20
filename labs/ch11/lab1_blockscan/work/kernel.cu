// Lab 11.1 starter (CUDA; also built for HIP and for the CPU emulators).
// One block scan, three ways. Book: sections 11.2 to 11.4, pp. 254-265.
//
//   provided  KoggeStoneKernel               Fig. 11.3, p. 256 (../../shared-g11/bs_two_barrier.cu)
//   (a)       KoggeStoneDoubleBufferKernel   Fig. 11.5, p. 260
//   (b)       warpScan                       Fig. 11.8, p. 264
//             blockScan                      Fig. 11.9, p. 264
//             WarpScanKernel                 Fig. 11.10, p. 265
//
// Contract: block b scans its own segment input[b*blockDim.x ... (b+1)*blockDim.x - 1],
// inclusive, with +, independently of every other block. Any n >= 1. A thread past n
// loads 0, the identity of +, takes part in every barrier and shuffle, and stores
// nothing. Block sizes are powers of two from 64 to 1024, chosen at run time by the
// test, so there is no SEG_SIZE: use blockDim.x, and size shared arrays for the largest
// block.
//
// Where the book's listings will not do as printed (the lesson pages say why):
//   - Fig. 11.8 passes the literal 0xffffffff and loops while stride < WARP_SIZE = 32.
//   - Fig. 11.9 declares its shared array of warp sums inside the device function, and
//     sizes it with a compile-time NUM_WARPS.
// Read ../../shared-g11/bs_two_barrier.cu first: (a) starts from it.

#define MAX_BLOCK_DIM 1024   // the largest block these kernels accept
#define MIN_WARP_SIZE 32     // the narrowest warp any backend here documents

// TODO 0: a type for the lane mask of __shfl_up_sync. CUDA's mask is an unsigned int.
//         HIP's _sync intrinsics refuse to compile unless the mask is a 64-bit integer,
//         in every warp width. __HIPCC__ is defined when HIP compiles this file.

// The book's helpers laneIdx() and warpIdx() (Chapter 10, pp. 238-239) use the constant
// WARP_SIZE = 32. Write them with the built-in warpSize instead.

// ---------------------------------------------------------------- (a)
__global__ void KoggeStoneDoubleBufferKernel(const float* input, float* output, unsigned int N) {
    // TODO 1: Fig. 11.5. Two shared buffers and two pointers to them, inBuffer_s and
    //         outBuffer_s, that swap roles every iteration.
    //   a. load this thread's element (or 0 past N) into the input buffer
    //   b. for strides 1, 2, 4, ... below blockDim.x, with ONE __syncthreads() per
    //      iteration: read the input buffer, write the output buffer, swap. Which
    //      threads add, and what must every other thread do so that the output buffer
    //      is complete?
    //   c. after the loop, which pointer holds the result? Store it if in range.
}

void launchKoggeStoneDB(const float* in, float* out, int n, int block) {
    // TODO 2: one block of `block` threads per segment of `block` elements, enough
    //         blocks to cover all n elements.
}

// ---------------------------------------------------------------- (b)
// Every lane of a warp calls this with its value and gets the inclusive scan of the
// warp's values back.
__device__ __forceinline__ float warpScan(float val) {
    // TODO 3: Fig. 11.8. Kogge-Stone again, but the value `stride` lanes to the left
    //         comes from __shfl_up_sync, with a full mask of your type from TODO 0.
    //         Strides run up to the warp's width, whatever it is. What does a lane below
    //         `stride` receive (ch11 facts A3), and what must it do with it?
    return val;
}

// Every thread of the block must call this. warpSums_s points to a shared array with
// room for one float per warp.
__device__ __forceinline__ float blockScan(float val, float* warpSums_s) {
    // TODO 4: Fig. 11.9, scan-scan-add over the warps.
    //   a. every warp scans its own values; the last lane of each warp holds its total
    //      and stores it in warpSums_s
    //   b. a barrier
    //   c. warp 0 scans the warp totals, one per lane. Lanes with no warp behind them
    //      must feed something harmless: what? Write the scanned totals back.
    //   d. a barrier
    //   e. every warp but the first adds the scanned total of the warps before it
    return val;
}

__global__ void WarpScanKernel(const float* input, float* output, unsigned int N) {
    // TODO 5: Fig. 11.10. Declare the shared array of warp totals here: how many entries
    //         does it need for any block size on any warp width? Load (0 past N), call
    //         blockScan from every thread, store if in range.
}

void launchWarpScan(const float* in, float* out, int n, int block) {
    // TODO 6: as TODO 2.
}
