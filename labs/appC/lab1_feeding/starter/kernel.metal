// Lab C.1 (Metal Shading Language). Provided complete; this lab's work is on the
// host side, in launch.mm. Not in the book: appendix C is about CUDA only.
//
// The same kernel as gatherAddKernel in kernel.cu: one thread per output element,
// out[r*cols + c] += table[ids[r]*cols + c]. encodeGatherAdd launches whole
// threadgroups, so threads past the end return. Sizes stay below 2^31 elements.
#include <metal_stdlib>
using namespace metal;

kernel void gather_add(device const float* table [[buffer(0)]],
                       device const int*   ids   [[buffer(1)]],
                       device float*       out   [[buffer(2)]],
                       constant uint&      count [[buffer(3)]],
                       constant uint&      cols  [[buffer(4)]],
                       uint i [[thread_position_in_grid]]) {
    if (i >= count * cols) return;
    uint r = i / cols;
    uint c = i % cols;
    out[i] += table[uint(ids[r]) * cols + c];
}
