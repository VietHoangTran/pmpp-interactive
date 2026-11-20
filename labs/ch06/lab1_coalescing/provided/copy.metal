// Provided reference for lab 6.1 (Metal): a plain copy, out[i] = in[i].
// Same kernel as provided/copy.cu. Its pipeline also reports this device's
// SIMD-group width (threadExecutionWidth), which the benchmark prints.
#include <metal_stdlib>
using namespace metal;

kernel void copy_floats(device const float* in  [[buffer(0)]],
                        device float*       out [[buffer(1)]],
                        constant int&       n   [[buffer(2)]],
                        uint3 gid [[thread_position_in_grid]]) {
    int i = int(gid.x);
    if (i < n) {
        out[i] = in[i];
    }
}
