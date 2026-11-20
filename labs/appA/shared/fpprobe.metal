// Lab A.1, provided (Metal Shading Language). Do not edit.
// The same three questions as fpprobe.cu, compiled the way the lab runtime compiles every
// kernel: default MTLCompileOptions, no math-mode pragma (notes/today/appA-facts.md C3). So
// the answers are those of Metal's default math mode on this device.
//
//   out[0] = in[0] + in[1]          1 + 3/4 ULP: next float above 1 to nearest, 1 toward zero
//   out[1] = in[2] * in[3]          2^-126 * 0.5: 2^-127 if denormals are kept, 0 if flushed
//   out[2] = in[4] * in[4] + in[5]  2^-24 if a * b + c was fused, 0 if not
#include <metal_stdlib>
using namespace metal;

kernel void fp_probe(device const float* in  [[buffer(0)]],
                     device float*       out [[buffer(1)]]) {
    out[0] = in[0] + in[1];
    out[1] = in[2] * in[3];
    out[2] = in[4] * in[4] + in[5];
}
