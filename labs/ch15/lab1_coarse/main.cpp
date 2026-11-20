// Lab 15.1 driver: tests and benchmarks the 128 x 128 coarsened kernel.
// You do not need to edit this file; your work goes in work/. The tests and the
// benchmark are shared by labs 15.1 to 15.5: see ../shared/stage_driver.hpp.
#include "../shared/stage_driver.hpp"

#if defined(PMPP_BACKEND_METAL)
// <variant>/launch.mm
void encodeGemmCoarse(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A,
                      id<MTLBuffer> B, id<MTLBuffer> C, uint32_t M, uint32_t K, uint32_t N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.path = PMPP_METAL_SOURCE;
  k.function = "gemm_coarse";
  k.encode = encodeGemmCoarse;
  return k;
}
#else
// <variant>/kernel.cu
void launchGemmCoarse(const float* A, const float* B, float* C, int M, int K, int N);
static ch15::Kernel stageKernel() {
  ch15::Kernel k;
  k.launch = launchGemmCoarse;
  return k;
}
#endif

int main(int argc, char** argv) {
  ch15::Stage st;
  st.lab = "15.1";
  st.title = "the 128 x 128 coarsened kernel (Figs. 15.3 to 15.7)";
  st.self = stageKernel();
  st.self.label = "15.1";
  // No previous stage: lab 5.3's kernel is the baseline, and the driver always times it.
  return ch15::stageMain(argc, argv, st);
}
