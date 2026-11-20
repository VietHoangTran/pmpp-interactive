// Lab 5.5 reference solution (Metal Shading Language).
// Metal's counterpart of `extern __shared__` is a kernel PARAMETER: a pointer in
// the threadgroup address space, bound with [[threadgroup(index)]]. The host sets
// its size in bytes with setThreadgroupMemoryLength:atIndex: before dispatching.
#include <metal_stdlib>
using namespace metal;

struct DynParams { uint width; uint tileWidth; };

kernel void tiled_matmul_dynamic(device const float* M [[buffer(0)]],
                                 device const float* N [[buffer(1)]],
                                 device float*       P [[buffer(2)]],
                                 constant DynParams& p [[buffer(3)]],
                                 threadgroup float* tiles [[threadgroup(0)]],
                                 uint3 tid [[thread_position_in_threadgroup]],
                                 uint3 bid [[threadgroup_position_in_grid]]) {
    const uint T = p.tileWidth, Width = p.width;
    threadgroup float* Mds = tiles;
    threadgroup float* Nds = tiles + T * T;      // elements, not bytes

    uint tx = tid.x, ty = tid.y;
    uint Row = bid.y * T + ty;
    uint Col = bid.x * T + tx;

    float Pvalue = 0;
    for (uint ph = 0; ph < (Width + T - 1) / T; ++ph) {
        uint mCol = ph * T + tx;
        uint nRow = ph * T + ty;
        Mds[ty * T + tx] = (Row < Width && mCol < Width) ? M[Row * Width + mCol] : 0.0f;
        Nds[ty * T + tx] = (nRow < Width && Col < Width) ? N[nRow * Width + Col] : 0.0f;
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (uint k = 0; k < T; ++k) {
            Pvalue += Mds[ty * T + k] * Nds[k * T + tx];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
}
