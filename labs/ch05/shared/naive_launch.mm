// Provided: encodes the naive baseline kernel on Metal.
#import <Metal/Metal.h>
#include <cstdint>

void encodeNaiveMatmul(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    const NSUInteger side = 16;
    NSUInteger groups = (width + side - 1) / side;
    [enc dispatchThreadgroups:MTLSizeMake(groups, groups, 1)
        threadsPerThreadgroup:MTLSizeMake(side, side, 1)];
}
