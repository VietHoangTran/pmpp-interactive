// Lab 4.4, provided and driver-side: you do not edit or call this.
// It asks the vendor's runtime how many blocks of YOUR kernel fit on one SM at
// a given block size, so the benchmark can set that beside your model. The
// runtime knows what your model has to be told: how many registers the
// compiled kernel really uses.
//
// CUDA: cudaOccupancyMaxActiveBlocksPerMultiprocessor (book p. 87).
// HIP:  the same call through pmpp/vendors/hip.h, which names
//       hipOccupancyMaxActiveBlocksPerMultiprocessor. On AMD the block size is
//       rounded up to whole wavefronts and the limits are wave slots, two
//       register pools and shared memory; there is no separate block-slot limit.
//
// UNVERIFIED on CUDA and HIP: written on a machine with neither. It refers to
// the kernel in your kernel.cu from another source file. If it does not build
// with your toolchain, configure once with
//   cmake -S labs -B labs/build -DPMPP_CH04_OCCUPANCY_API=OFF
// and the benchmark runs without the vendor's column.

// Defined in <variant>/kernel.cu. The signature must match it exactly.
__global__ void gatherSumKernel(const float* table, const int* idx, float* out, int n, int k);

// Blocks of gatherSumKernel per SM at this block size, or -1 if the runtime
// would not say.
int vendorBlocksPerSm(int threadsPerBlock) {
#if defined(PMPP_EMU)
    // The emulator's occupancy function returns a made-up number that exists
    // only to size cooperative launches. It must never be shown as occupancy.
    (void)threadsPerBlock;
    return -1;
#else
    int numBlocks = 0;
    if (cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, gatherSumKernel, threadsPerBlock, 0) != cudaSuccess) {
        return -1;
    }
    return numBlocks;
#endif
}
