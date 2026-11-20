// Lab 8.1 starter: the Metal counterparts of the two CUDA <<<grid, block>>> launches.
#import <Metal/Metal.h>
#include <cstdint>

#include "../stencil.h"

void encodeStencilBasic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t N, StencilCoeffs c) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(N) atIndex:2];
    [enc setBytes:&c length:sizeof(c) atIndex:3];

    // TODO: 8 x 8 x 8 threadgroups, enough of them to give every grid point its thread:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, ?) threadsPerThreadgroup:MTLSizeMake(?, ?, ?)];
}

void encodeStencilRegTiled(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t N, uint32_t depth,
                           uint32_t tileWidth, StencilCoeffs c) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(N) atIndex:2];
    [enc setBytes:&depth length:sizeof(depth) atIndex:3];
    [enc setBytes:&c length:sizeof(c) atIndex:4];

    // TODO: tileWidth x tileWidth threadgroups (tileWidth is the TILE_WIDTH the kernel
    //       was compiled with). In x and y, one threadgroup per (tileWidth - 2)-wide
    //       output tile; in z, one per slab of `depth` planes.
}
