// Lab 6.3 reference solution (Metal Shading Language).
// The harness defines TILE_WIDTH when it compiles this file (16, and 32 where the
// pipeline accepts 1024 threads per threadgroup).
//
// Apple documents no bank structure for threadgroup memory (notes/today/ch06-facts.md C2,
// infra-facts C13). The only access-order advice Apple gives for it is to have
// neighbouring threads of a quad group touch neighbouring elements. Whether pitch
// TILE_WIDTH or TILE_WIDTH + 1 is faster here is something only a measurement says.
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

kernel void tiled_transpose(device const float* in    [[buffer(0)]],
                            device float*       out   [[buffer(1)]],
                            constant int&       rows  [[buffer(2)]],
                            constant int&       cols  [[buffer(3)]],
                            constant int&       pitch [[buffer(4)]],
                            uint2 tid [[thread_position_in_threadgroup]],
                            uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float tile[TILE_WIDTH * (TILE_WIDTH + 1)];

    int tx = int(tid.x), ty = int(tid.y);

    int r = int(bid.y) * TILE_WIDTH + ty;
    int c = int(bid.x) * TILE_WIDTH + tx;
    if (r < rows && c < cols) {
        tile[ty * pitch + tx] = in[r * cols + c];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    int outRow = int(bid.x) * TILE_WIDTH + ty;
    int outCol = int(bid.y) * TILE_WIDTH + tx;
    if (outRow < cols && outCol < rows) {
        out[outRow * rows + outCol] = tile[tx * pitch + ty];
    }
}
