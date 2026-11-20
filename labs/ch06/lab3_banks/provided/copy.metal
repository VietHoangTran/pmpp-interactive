// Provided instrument for lab 6.3 on Metal: the plain copy of provided/copy.cu, the
// yardstick for a transpose. Do not edit.
#include <metal_stdlib>
using namespace metal;

kernel void copy_floats(device const float* in  [[buffer(0)]],
                        device float*       out [[buffer(1)]],
                        constant int&       n   [[buffer(2)]],
                        uint gid [[thread_position_in_grid]]) {
    int i = int(gid);
    if (i < n) out[i] = in[i];
}
