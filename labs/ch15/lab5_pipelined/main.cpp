// Lab 15.5 driver: tests and benchmarks the double-buffered, software-pipelined kernel.
// You do not need to edit this file; your work goes in work/. The tests and the
// benchmark are shared by labs 15.1 to 15.5: see ../shared/stage_driver.hpp.
// The tests run the layout for a warp (SIMD-group) of 32 and of 64 threads on every
// backend, and on CUDA and HIP also the launch that reads the device's width.
#include "../shared/stage_driver.hpp"

#if defined(PMPP_BACKEND_METAL)
// <variant>/launch.mm
void encodeGemmPipelined(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                         id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
// Lab 15.4's solution, copied into the build by ../shared/ch15.cmake.
void encodeGemmPadded(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                      id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.path = PMPP_METAL_SOURCE;
  k.function = "gemm_pipelined";
  k.encode = encodeGemmPipelined;
  k.simdWidth = true;   // compiled with SIMD_WIDTH=<w>
  return k;
}
static ch15::Kernel previousKernel() {
  ch15::Kernel k;
  k.path = PMPP_CH15_PREV_METAL;
  k.function = "gemm_padded";
  k.encode = encodeGemmPadded;
  k.simdWidth = true;
  return k;
}
#else
// <variant>/kernel.cu
void launchGemmPipelined(const float* A, const float* B, float* C, int M, int K, int N);
void launchGemmPipelinedWidth(const float* A, const float* B, float* C, int M, int K, int N, int width);
// Lab 15.4's solution, copied into the build by ../shared/ch15.cmake.
void launchGemmPadded(const float* A, const float* B, float* C, int M, int K, int N);
void launchGemmPaddedWidth(const float* A, const float* B, float* C, int M, int K, int N, int width);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.launch = launchGemmPipelined;
  k.launchWidth = launchGemmPipelinedWidth;
  return k;
}
static ch15::Kernel previousKernel() {
  ch15::Kernel k;
  k.launch = launchGemmPadded;
  k.launchWidth = launchGemmPaddedWidth;
  return k;
}
#endif

int main(int argc, char** argv) {
  ch15::Stage st;
  st.lab = "15.5";
  st.title = "double buffering and software pipelining (section 15.8, Fig. 15.14)";
  st.self = stageKernel();
  st.self.label = "15.5";
  st.previous = previousKernel();
  st.previous.label = "15.4";
  // A missing or misplaced barrier is a race: on a GPU it may pass one launch and fail the
  // next, so every case is launched 10 times there (the emulators need one run each).
  st.gpuLaunches = 10;
  return ch15::stageMain(argc, argv, st);
}
