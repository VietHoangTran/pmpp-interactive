// Lab 6.5: the Metal counterpart of launchDoubleBufferedMatmul. The dispatch is
// lab 5.2's and is correct: double buffering changes the kernel, not the launch.
#import <Metal/Metal.h>
#include <cstdint>

void encodeDoubleBufferedMatmul(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P,
                                uint32_t width, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];

    // One threadgroup per output tile, one thread per element of the tile.
    MTLSize threadsPerGroup = MTLSizeMake(tileWidth, tileWidth, 1);
    MTLSize groups = MTLSizeMake(width / tileWidth, width / tileWidth, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
