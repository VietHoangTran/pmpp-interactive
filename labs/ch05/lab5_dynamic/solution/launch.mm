// Lab 5.5 reference solution: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

struct DynParams { uint32_t width, tileWidth; };

void encodeTiledMatmulDynamic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                              id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P,
                              uint32_t width, uint32_t tileWidth) {
    DynParams p = {width, tileWidth};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];

    // Both tiles, in bytes. Metal requires a multiple of 16.
    NSUInteger bytes = 2 * (NSUInteger)tileWidth * tileWidth * sizeof(float);
    bytes = (bytes + 15) / 16 * 16;
    [enc setThreadgroupMemoryLength:bytes atIndex:0];   // matches [[threadgroup(0)]] in the kernel

    NSUInteger groups = (width + tileWidth - 1) / tileWidth;
    [enc dispatchThreadgroups:MTLSizeMake(groups, groups, 1)
        threadsPerThreadgroup:MTLSizeMake(tileWidth, tileWidth, 1)];
}
