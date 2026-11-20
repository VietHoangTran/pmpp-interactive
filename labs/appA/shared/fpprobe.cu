// Lab A.1, provided (CUDA; also built for HIP and for the CPU emulators). Do not edit.
//
// Three questions about this device's float arithmetic, asked with real instructions. The
// inputs come from a buffer the driver fills, so the compiler cannot work the answers out
// while it builds the kernel (notes/today/appA-facts.md E1). Not in the book.
//
//   out[0] = in[0] + in[1]          1 + 3/4 of an ULP of 1. Rounded to nearest: the next
//                                   float above 1. Rounded toward zero: 1. (A.4, Exercise 3;
//                                   Metal allows either mode, C2.)
//   out[1] = in[2] * in[3]          the smallest normal float, 2^-126, halved: the subnormal
//                                   2^-127 if denormals are kept, 0 if they are flushed (A6, C2).
//   out[2] = in[4] * in[4] + in[5]  (1 + 2^-12)^2 - (1 + 2^-11) is exactly 2^-24. With one
//                                   rounding (a fused multiply-add) that is the result; with
//                                   the product rounded first it is 0 (A9).
//
// On the emulator the kernel runs on the host CPU, so the answers describe the CPU and the
// host compiler's choices, not a GPU. On CUDA and HIP the answers are those of this file's
// build flags (nvcc and HIP-Clang contract a * b + c by default, A9 and B5).
#include "../lab1_sumerror/sumerror.h"

__global__ void fpProbeKernel(const float* in, float* out) {
    out[0] = in[0] + in[1];
    out[1] = in[2] * in[3];
    out[2] = in[4] * in[4] + in[5];
}

void launchFpProbe(const float* in, float* out) {
    fpProbeKernel<<<1, 1>>>(in, out);
}
