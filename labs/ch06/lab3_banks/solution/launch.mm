// Lab 6.3 reference solution: the Metal counterpart of launchTiledTranspose.
#import <Metal/Metal.h>
#include <cstdint>

void encodeTiledTranspose(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> in, id<MTLBuffer> out,
                          int32_t rows, int32_t cols, int32_t pitch, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&pitch length:sizeof(pitch) atIndex:4];

    // One threadgroup per tile, rounded up: x along the input's columns, y along its rows.
    MTLSize groups = MTLSizeMake(((NSUInteger)cols + tileWidth - 1) / tileWidth,
                                 ((NSUInteger)rows + tileWidth - 1) / tileWidth, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(tileWidth, tileWidth, 1)];
}
