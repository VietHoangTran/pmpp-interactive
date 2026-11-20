// Lab 15.3 driver: tests and benchmarks the re-mapped kernel with float4 stores.
// You do not need to edit this file; your work goes in work/. The tests and the
// benchmark are shared by labs 15.1 to 15.5: see ../shared/stage_driver.hpp.
// The tests run the layout for a warp (SIMD-group) of 32 and of 64 threads on every
// backend, and on CUDA and HIP also the launch that reads the device's width.
#include "../shared/stage_driver.hpp"

#if defined(PMPP_BACKEND_METAL)
// <variant>/launch.mm
void encodeGemmRemap(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                     id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
// Lab 15.2's solution, copied into the build by ../shared/ch15.cmake.
void encodeGemmRegTile(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                       id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.path = PMPP_METAL_SOURCE;
  k.function = "gemm_remap";
  k.encode = encodeGemmRemap;
  k.simdWidth = true;   // compiled with SIMD_WIDTH=<w>
  return k;
}
static ch15::Kernel previousKernel() {
  ch15::Kernel k;
  k.path = PMPP_CH15_PREV_METAL;
  k.function = "gemm_regtile";
  k.encode = encodeGemmRegTile;
  return k;
}
#else
// <variant>/kernel.cu
void launchGemmRemap(const float* A, const float* B, float* C, int M, int K, int N);
void launchGemmRemapWidth(const float* A, const float* B, float* C, int M, int K, int N, int width);
// Lab 15.2's solution, copied into the build by ../shared/ch15.cmake.
void launchGemmRegTile(const float* A, const float* B, float* C, int M, int K, int N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.launch = launchGemmRemap;
  k.launchWidth = launchGemmRemapWidth;
  return k;
}
static ch15::Kernel previousKernel() {
  ch15::Kernel k;
  k.launch = launchGemmRegTile;
  return k;
}
#endif

int main(int argc, char** argv) {
  ch15::Stage st;
  st.lab = "15.3";
  st.title = "four 4 x 4 sub-tiles per thread and float4 stores (section 15.5, Exercises 3 and 2)";
  st.self = stageKernel();
  st.self.label = "15.3";
  st.previous = previousKernel();
  st.previous.label = "15.2";
  return ch15::stageMain(argc, argv, st);
}
