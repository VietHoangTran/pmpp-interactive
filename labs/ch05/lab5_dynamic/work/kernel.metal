// Lab 5.5 starter (Metal Shading Language).
// Metal's counterpart of `extern __shared__` is a kernel PARAMETER: a pointer in
// the threadgroup address space, bound with [[threadgroup(index)]]. The host sets
// its size in bytes with setThreadgroupMemoryLength:atIndex: before dispatching.
#include <metal_stdlib>
using namespace metal;

struct DynParams { uint width; uint tileWidth; };   // must match launch.mm

kernel void tiled_matmul_dynamic(device const float* M [[buffer(0)]],
                                 device const float* N [[buffer(1)]],
                                 device float*       P [[buffer(2)]],
                                 constant DynParams& p [[buffer(3)]],
                                 threadgroup float* tiles [[threadgroup(0)]],
                                 uint3 tid [[thread_position_in_threadgroup]],
                                 uint3 bid [[threadgroup_position_in_grid]]) {
    // TODO: split `tiles` into the two tiles and run the tiled algorithm.
}
