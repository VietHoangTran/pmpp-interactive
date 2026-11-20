// Lab 5.4: Metal host side (provided; the defects are all in the kernel).
#import <Metal/Metal.h>
#include <cstdint>

void encodeTiledMatmul(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P,
                       uint32_t width, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    NSUInteger groups = (width + tileWidth - 1) / tileWidth;
    [enc dispatchThreadgroups:MTLSizeMake(groups, groups, 1)
        threadsPerThreadgroup:MTLSizeMake(tileWidth, tileWidth, 1)];
}
