// Lab 8.1 reference solution: the Metal counterparts of the two CUDA launches.
#import <Metal/Metal.h>
#include <cstdint>

#include "../stencil.h"

// Fig. 8.6: 8 x 8 x 8 threadgroups (exercise 1b), enough to cover every grid point.
void encodeStencilBasic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t N, StencilCoeffs c) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(N) atIndex:2];
    [enc setBytes:&c length:sizeof(c) atIndex:3];

    const NSUInteger side = 8;
    const NSUInteger groups = (N + side - 1) / side;
    [enc dispatchThreadgroups:MTLSizeMake(groups, groups, groups)
        threadsPerThreadgroup:MTLSizeMake(side, side, side)];
}

// Fig. 8.12: tileWidth x tileWidth threadgroups; in x and y one group per
// (tileWidth - 2)-wide output tile, in z one group per slab of `depth` planes.
void encodeStencilRegTiled(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t N, uint32_t depth,
                           uint32_t tileWidth, StencilCoeffs c) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&N length:sizeof(N) atIndex:2];
    [enc setBytes:&depth length:sizeof(depth) atIndex:3];
    [enc setBytes:&c length:sizeof(c) atIndex:4];

    const NSUInteger outTile = tileWidth - 2;
    const NSUInteger tiles = (N + outTile - 1) / outTile;
    const NSUInteger slabs = (N + depth - 1) / depth;
    [enc dispatchThreadgroups:MTLSizeMake(tiles, tiles, slabs)
        threadsPerThreadgroup:MTLSizeMake(tileWidth, tileWidth, 1)];
}
