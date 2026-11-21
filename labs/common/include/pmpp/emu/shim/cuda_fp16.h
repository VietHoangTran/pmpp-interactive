// Stands in for <cuda_fp16.h> in emulator builds and in the CUDA front-end check. Under HIP
// the header is <hip/hip_fp16.h> (infra-facts B12): a .cu file that must also build for HIP
// chooses the include with #if.
#pragma once
#if defined(PMPP_CUDA_FRONTEND_CHECK)
#define PMPP_STUB_WANT_FP16
#include "pmpp/check/cuda_stub.h"
#else
#include "pmpp/emu/cuda_fp16_emu.h"
#endif
