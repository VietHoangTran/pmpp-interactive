// The kernel staged_loop_metal.mm dispatches: the same stand-in as chunkKernel in
// staged_loop.cu, one thread per element of a chunk.
#include <metal_stdlib>
using namespace metal;

kernel void chunkKernel(device const float* in [[buffer(0)]], device float* out [[buffer(1)]],
                        constant int& len [[buffer(2)]], uint i [[thread_position_in_grid]]) {
    if (int(i) < len) out[i] = 2.0f * in[i] + 1.0f;
}
