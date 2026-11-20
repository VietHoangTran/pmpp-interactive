// Labs 15.1 to 15.5, provided and driver-side: you do not edit or call this.
// Asks the CUDA or HIP runtime how many blocks of this stage's kernel fit on one SM at
// 256 threads per block (cudaOccupancyMaxActiveBlocksPerMultiprocessor, the call of
// lab 4.4; notes/today/ch04-facts.md A14). The runtime knows what no model can: how many
// registers the compiled kernel really uses, and the allocation granularity NVIDIA does
// not publish (ch15-facts.md A3). The book's own estimate is one block per SM (p. 363).
//
// The build names the kernel (labs/ch15/shared/ch15.cmake): PMPP_CH15_KERNEL, and
// PMPP_CH15_KERNEL_TEMPLATE = 1 for the stages whose kernel is a template over the warp
// width (labs 15.3 to 15.5). Here they are declared only; the definitions are in your
// kernel.cu, and for the templates the instances are the ones its launch creates.
//
// UNVERIFIED on CUDA and HIP: written on a machine with neither. Referring to a kernel,
// and to a kernel template's instance, defined in another source file has not been built
// by nvcc or a HIP compiler. If it does not build, configure once with
//   cmake -S labs -B <your build directory> -DPMPP_CH15_OCCUPANCY_API=OFF
// and the benchmark prints no occupancy row.
#ifndef PMPP_CH15_KERNEL
// Defaults for the CUDA front-end check, which parses this file with no build definitions.
#define PMPP_CH15_KERNEL gemmCoarseKernel
#define PMPP_CH15_KERNEL_TEMPLATE 0
#endif

#if PMPP_CH15_KERNEL_TEMPLATE
template <int W>
__global__ void PMPP_CH15_KERNEL(const float* A, const float* B, float* C, int M, int K, int N);
#else
__global__ void PMPP_CH15_KERNEL(const float* A, const float* B, float* C, int M, int K, int N);
#endif

// Blocks of this stage's kernel per SM, or -1 if the runtime would not say. `width` is
// the warp width whose instance to ask about (ignored when the kernel is not a template).
int ch15StageBlocksPerSm(int width, int threadsPerBlock) {
#if defined(PMPP_EMU) || defined(PMPP_CH15_NO_OCCUPANCY)
    // The emulator's occupancy function returns a made-up number that exists only to
    // size cooperative launches. It must never be shown as occupancy.
    (void)width;
    (void)threadsPerBlock;
    return -1;
#else
    int numBlocks = 0;
    cudaError_t e;
#if PMPP_CH15_KERNEL_TEMPLATE
    // The two instances the stages' launch functions create (warp widths of NVIDIA and
    // AMD RDNA, and of AMD CDNA; ch15-facts B1).
    if (width == 32) {
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, PMPP_CH15_KERNEL<32>, threadsPerBlock, 0);
    } else if (width == 64) {
        e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, PMPP_CH15_KERNEL<64>, threadsPerBlock, 0);
    } else {
        return -1;
    }
#else
    (void)width;
    e = cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks, PMPP_CH15_KERNEL, threadsPerBlock, 0);
#endif
    return e == cudaSuccess ? numBlocks : -1;
#endif
}
