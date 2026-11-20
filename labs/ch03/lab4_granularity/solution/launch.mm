// Lab 3.4 reference solution: Metal host side.
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
    // A 1D grid still takes three sizes: the unused height and depth are 1.
    const NSUInteger perGroup = 256;
    [enc dispatchThreadgroups:MTLSizeMake((width + perGroup - 1) / perGroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

// ---------- exercise 1b: a 1D grid, one thread per column ----------
// The dispatch is the same as encodeRow's. What a thread's position means is
// decided by the kernel bound to the pipeline, not by the launch.
void encodeCol(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
               id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    bind(enc, pipeline, M, N, P, width);
    const NSUInteger perGroup = 256;
    [enc dispatchThreadgroups:MTLSizeMake((width + perGroup - 1) / perGroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}
