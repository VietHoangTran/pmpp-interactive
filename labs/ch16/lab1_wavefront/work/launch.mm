// Lab 16.1 starter: the Metal host side.
//
// Each encode function writes one dispatch per front of tiles into the compute
// encoder the driver opened (and will close, commit and wait for). That encoder runs
// its dispatches in order, and buffers made from the device are hazard-tracked, so a
// dispatch sees what the earlier ones stored: the counterpart of the book's kernel
// relaunch between fronts (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <algorithm>

struct SwParams { int L; int d; int firstRow; int stride; };   // must match kernel.metal

static void bindTable(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:b offset:0 atIndex:2];
}

// Square tiles: 2T - 1 fronts; front d holds the tiles (row, d - row).
void encodeSwSquare(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b, int Lseq, int W, int pad) {
    bindTable(enc, pipeline, H, a, b);
    // TODO: for each front:
    //   1. its SwParams, with setBytes:length:atIndex:3
    //   2. the tile's size in bytes, rounded up to a multiple of 16 (Metal's rule), with
    //        [enc setThreadgroupMemoryLength:? atIndex:0];    // [[threadgroup(0)]]
    //   3. one threadgroup of W threads per tile of the front:
    //        [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

// Sheared tiles: 3T - 1 fronts.
void encodeSwHyper(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b, int Lseq, int W, int pad) {
    bindTable(enc, pipeline, H, a, b);
    // TODO: the same three steps, with the fronts of sheared tiles.
}
