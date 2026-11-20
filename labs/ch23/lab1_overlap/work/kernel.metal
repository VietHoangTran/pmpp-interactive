// Lab 23.1 (Metal Shading Language). Provided complete; this lab's work is on the
// host side, in launch.mm. Not in the book: chapter 23 is about CUDA only.
//
// The same kernel as polyKernel in kernel.cu: one thread per element,
// out[i] = p(in[i]), with p the polynomial of overlap.h (a_k = 0.5 for even k,
// -0.25 for odd k) evaluated by Horner's rule, `degree` multiply-adds per
// element. encodePoly launches whole threadgroups, so threads past the end return.
#include <metal_stdlib>
using namespace metal;

kernel void poly_kernel(device const float* in     [[buffer(0)]],
                        device float*       out    [[buffer(1)]],
                        constant uint&      count  [[buffer(2)]],
                        constant uint&      degree [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    const float x = in[i];
    float y = (degree & 1u) ? -0.25f : 0.5f;           // a_degree
    for (uint k = degree; k > 0; --k)
        y = y * x + (((k - 1u) & 1u) ? -0.25f : 0.5f);  // y = y*x + a_(k-1)
    out[i] = y;
}
