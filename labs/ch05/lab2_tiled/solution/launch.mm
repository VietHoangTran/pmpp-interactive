// Lab 5.2 reference solution: the Metal counterpart of a CUDA <<<grid, block>>> launch.
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

    // "block" = threadgroup. One threadgroup per output tile, one thread per element.
    MTLSize threadsPerGroup = MTLSizeMake(tileWidth, tileWidth, 1);
    MTLSize groups = MTLSizeMake(width / tileWidth, width / tileWidth, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
