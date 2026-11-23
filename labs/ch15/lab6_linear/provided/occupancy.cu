// Lab 15.6, provided: what the vendor's runtime says about YOUR compiled M3 kernel, for the
// benchmark's occupancy row. You do not edit or call this.
//
// Section 15.7 (pp. 363-364) argues that a kernel like this one runs at low occupancy on
// purpose: about 80 registers of tiles per thread, in practice up to the 255 limit, so one
// 256-thread block per SM. The runtime knows what the argument has to assume:
//   - registers per thread: cudaFuncGetAttributes(...).numRegs (ch04-facts A13; on HIP,
//     hipFuncAttributes.numRegs counts vector registers only, ch04-facts B10);
//   - blocks per SM: cudaOccupancyMaxActiveBlocksPerMultiprocessor (ch04-facts A14), which
//     accounts for the register allocation granularity NVIDIA does not publish
//     (ch15-facts A3).
//
// It refers to fastLinearKernel<32> and <64>, which your kernel.cu instantiates in
// launchLinearWidth. The declaration below must match your kernel exactly.
//
// Only CUDA and HIP builds ask the runtime: the emulators' occupancy function returns a
// made-up number, which must never be shown as occupancy, and the front-end check has no
// runtime. Elsewhere the functions return -1.
//
// UNVERIFIED on CUDA and HIP: written on a machine with neither. If it does not build,
// configure with -DPMPP_CH15_OCCUPANCY_API=OFF and the benchmark runs without the row.
#if defined(PMPP_CH15_OCCUPANCY_API) && (defined(PMPP_BACKEND_CUDA) || defined(PMPP_BACKEND_HIP))

#if defined(PMPP_BACKEND_HIP)
#define cudaFuncAttributes    hipFuncAttributes
#define cudaFuncGetAttributes hipFuncGetAttributes
#endif

template <int WARP>
__global__ void fastLinearKernel(const float* X, const float* W, float* Y, int batch, int in, int out);

// Registers per thread of the kernel built for this warp width, or -1.
int m3RegistersPerThread(int width) {
    cudaFuncAttributes attr;
    const cudaError_t e = width == 64 ? cudaFuncGetAttributes(&attr, (const void*)fastLinearKernel<64>)
                                      : cudaFuncGetAttributes(&attr, (const void*)fastLinearKernel<32>);
    return e == cudaSuccess ? attr.numRegs : -1;
}

// Resident 256-thread blocks per SM (CU on AMD), or -1.
int m3BlocksPerSm(int width) {
    int blocks = 0;
    const cudaError_t e =
        width == 64 ? cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fastLinearKernel<64>, 256, 0)
                    : cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks, fastLinearKernel<32>, 256, 0);
    return e == cudaSuccess ? blocks : -1;
}

#else

int m3RegistersPerThread(int) { return -1; }
int m3BlocksPerSm(int) { return -1; }

#endif
