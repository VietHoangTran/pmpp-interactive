// Lab 16.1 reference solution: the Metal host side.
//
// One dispatch per front of tiles, all in the compute encoder the driver opened and
// will close, commit and wait for. That encoder runs its dispatches in order, and
// buffers made from the device are hazard-tracked, so each front sees what the earlier
// fronts stored (notes/today/ch02-facts.md C9, ch16-facts.md C4). This is the book's
// kernel relaunch between fronts (p. 384), in Metal's terms.
#import <Metal/Metal.h>
#include <algorithm>

struct SwParams { int L; int d; int firstRow; int stride; };   // as in kernel.metal

// W rows of `stride` ints, rounded up: Metal wants a multiple of 16 bytes
// (notes/today/ch05-facts.md C8).
static NSUInteger tileBytes(int W, int stride) {
    const NSUInteger bytes = (NSUInteger)W * stride * sizeof(int);
    return (bytes + 15) / 16 * 16;
}

static void bindTable(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:b offset:0 atIndex:2];
}

static void dispatchFront(id<MTLComputeCommandEncoder> enc, SwParams p, int firstRow, int lastRow, int W) {
    p.firstRow = firstRow;
    [enc setBytes:&p length:sizeof(p) atIndex:3];
    [enc setThreadgroupMemoryLength:tileBytes(W, p.stride) atIndex:0];   // [[threadgroup(0)]]
    [enc dispatchThreadgroups:MTLSizeMake(lastRow - firstRow + 1, 1, 1) threadsPerThreadgroup:MTLSizeMake(W, 1, 1)];
}

// Front d holds the tiles (row, d - row): 2T - 1 dispatches.
void encodeSwSquare(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b, int Lseq, int W, int pad) {
    const int T = (Lseq + W - 1) / W;
    bindTable(enc, pipeline, H, a, b);
    for (int d = 0; d < 2 * T - 1; d++) {
        SwParams p = {Lseq + 1, d, 0, W + pad};
        dispatchFront(enc, p, std::max(0, d - (T - 1)), std::min(d, T - 1), W);
    }
}

// Front d holds the tiles (row, d - 2 * row) with 0 <= d - 2 * row <= T: 3T - 1 dispatches.
void encodeSwHyper(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b, int Lseq, int W, int pad) {
    const int T = (Lseq + W - 1) / W;
    bindTable(enc, pipeline, H, a, b);
    for (int d = 0; d < 3 * T - 1; d++) {
        SwParams p = {Lseq + 1, d, 0, W + pad};
        dispatchFront(enc, p, std::max(0, (d - T + 1) / 2), std::min(T - 1, d / 2), W);
    }
}
