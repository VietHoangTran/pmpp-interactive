// Lab A.1, provided (do not edit): what the driver, your kernel.cu and the device probe share.
//
// Included by main.cpp (plain C++), by work/kernel.cu and solution/kernel.cu (CUDA, HIP and
// both emulators) and by ../shared/fpprobe.cu. Everything that names a device type or a
// device function is guarded, so main.cpp sees only the enum and the prototypes.
#pragma once

// Which accumulator a launcher should use. The data are always float32 in memory.
enum Acc { AccFloat = 0, AccHalf = 1, AccBfloat = 2 };

// Your launchers (work/kernel.cu). Each returns false, without launching anything, when this
// build has no kernel for `acc` (FP16 and BF16 exist on real CUDA and HIP devices only), or
// when `block` or `coarse` is outside what the kernels accept.
//   x         n float32 values on the device
//   partials  one float per block of the first pass: at least ceil(n / block) floats
//   out       one float: the sum
bool launchSumSerial(const float* x, float* out, int n, Acc acc);
bool launchSumTree(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc);
bool launchSumKahan(const float* x, float* partials, float* out, int n, int coarse, int block, Acc acc);

// The device probe (../shared/fpprobe.cu). in: the 6 inputs the driver writes, out: 3 results.
void launchFpProbe(const float* in, float* out);

// ---------------------------------------------------------------- device side only
// __fadd_rn and __fsub_rn: CUDA's float addition and subtraction that "the compiler never
// merges into the FFMA or DFMA instructions" (notes/today/appA-facts.md A9, A13). HIP has
// both under the same names (B4). The CPU emulator has neither, and the CUDA front-end
// check (labs/pmpp.py) declares no CUDA Math API functions, so:
//  - on the emulator they are plain + and - on the host CPU, which the host compiler does
//    not reassociate by default (B5);
//  - for the front-end check they are declared only. nvcc and hipcc never see either.
#if defined(PMPP_EMU)
__device__ inline float __fadd_rn(float x, float y) { return x + y; }
__device__ inline float __fsub_rn(float x, float y) { return x - y; }
#elif defined(PMPP_CUDA_FRONTEND_CHECK)
__device__ float __fadd_rn(float x, float y);
__device__ float __fsub_rn(float x, float y);
#endif

// The 16-bit accumulators exist on real CUDA and HIP builds only. The emulators do not
// model __half or __nv_bfloat16, and the front-end check's stub has neither (appA-facts E2).
// Name them only inside `#if PMPP_NARROW_ACC`.
//  - CUDA: __half from <cuda_fp16.h>; __nv_bfloat16 from <cuda_bf16.h>, which needs compute
//    capability 8.0 or higher (A2, A16). The driver checks the capability before it asks
//    for a BF16 sum.
//  - HIP: __half from <hip/hip_fp16.h>; the BF16 type is __hip_bfloat16 from
//    <hip/hip_bf16.h> (B1), given CUDA's name here so one kernel.cu serves both.
// Neither branch has been compiled here: no CUDA or ROCm toolkit on the machine this lab
// was written on. Unverified.
#if (defined(__CUDACC__) || defined(__HIPCC__)) && !defined(PMPP_EMU) && !defined(PMPP_CUDA_FRONTEND_CHECK)
#define PMPP_NARROW_ACC 1
#if defined(__HIPCC__)
#include <hip/hip_fp16.h>
#include <hip/hip_bf16.h>
typedef __hip_bfloat16 __nv_bfloat16;
#else
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif
#else
#define PMPP_NARROW_ACC 0
#endif
