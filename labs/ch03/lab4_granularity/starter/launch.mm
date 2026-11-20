// Lab 3.4 starter: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

static void bind(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
}

// ---------- provided: a 2D grid of 16 x 16 threadgroups, one thread per element ----------
void encodeElement(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    bind(enc, pipeline, M, N, P, width);
    const NSUInteger side = 16;
    NSUInteger groups = (width + side - 1) / side;
    [enc dispatchThreadgroups:MTLSizeMake(groups, groups, 1)
        threadsPerThreadgroup:MTLSizeMake(side, side, 1)];
}

// ---------- exercise 1a: a 1D grid, one thread per row ----------
void encodeRow(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
               id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    bind(enc, pipeline, M, N, P, width);
    // TODO: dispatch a 1D grid of 256-thread threadgroups with one thread per row,
    //       for any width. MTLSize has no defaults: the sizes you do not use must be 1.
}

// ---------- exercise 1b: a 1D grid, one thread per column ----------
void encodeCol(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
               id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    bind(enc, pipeline, M, N, P, width);
    // TODO: the same, with one thread per column.
}
