// Lab 15.2 driver: tests and benchmarks the register-tiled kernel.
// You do not need to edit this file; your work goes in work/. The tests and the
// benchmark are shared by labs 15.1 to 15.5: see ../shared/stage_driver.hpp.
#include "../shared/stage_driver.hpp"

#if defined(PMPP_BACKEND_METAL)
// <variant>/launch.mm
void encodeGemmRegTile(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                       id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
// Lab 15.1's solution, copied into the build by ../shared/ch15.cmake.
void encodeGemmCoarse(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                      id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.path = PMPP_METAL_SOURCE;
  k.function = "gemm_regtile";
  k.encode = encodeGemmRegTile;
  return k;
}
static ch15::Kernel previousKernel() {
  ch15::Kernel k;
  k.path = PMPP_CH15_PREV_METAL;
  k.function = "gemm_coarse";
  k.encode = encodeGemmCoarse;
  return k;
}
#else
// <variant>/kernel.cu
void launchGemmRegTile(const float* A, const float* B, float* C, int M, int K, int N);
// Lab 15.1's solution, copied into the build by ../shared/ch15.cmake.
void launchGemmCoarse(const float* A, const float* B, float* C, int M, int K, int N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.launch = launchGemmRegTile;
  return k;
}
static ch15::Kernel previousKernel() {
  ch15::Kernel k;
  k.launch = launchGemmCoarse;
  return k;
}
#endif

int main(int argc, char** argv) {
  ch15::Stage st;
  st.lab = "15.2";
  st.title = "register tiling: the loops of mm interchanged (Fig. 15.9)";
  st.self = stageKernel();
  st.self.label = "15.2";
  st.previous = previousKernel();
  st.previous.label = "15.1";
  return ch15::stageMain(argc, argv, st);
}
