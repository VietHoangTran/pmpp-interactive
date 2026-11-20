// Provided: encodes the two-barrier baseline kernel on Metal (the same dispatch as
// lab 5.2: one threadgroup per output tile, one thread per element of the tile).
#import <Metal/Metal.h>
#include <cstdint>

void encodeTwoBarrierMatmul(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P,
                            uint32_t width, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(width / tileWidth, width / tileWidth, 1)
        threadsPerThreadgroup:MTLSizeMake(tileWidth, tileWidth, 1)];
}
