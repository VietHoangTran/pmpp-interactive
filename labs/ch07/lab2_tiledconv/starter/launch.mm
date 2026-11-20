// Lab 7.2 starter: the Metal counterparts of launchHalo and launchCached.
// The pipeline was compiled with this tile width and radius as macros; they are
// passed here as well so that the grid can be computed.
#import <Metal/Metal.h>
#include <cstdint>

static void bindConvolution(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                            int32_t width, int32_t height) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:N offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc setBytes:&height length:sizeof(height) atIndex:4];
}

// @part halo
void encodeConvTiledHalo(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                         int32_t width, int32_t height, uint32_t tile, uint32_t radius) {
    bindConvolution(enc, pipeline, N, F, P, width, height);
    // TODO: the same block and grid as launchHalo in kernel.cu ("block" = threadgroup):
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, ?, 1)];
}

// @part cached
void encodeConvTiledCached(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                           int32_t width, int32_t height, uint32_t tile, uint32_t /*radius*/) {
    bindConvolution(enc, pipeline, N, F, P, width, height);
    MTLSize threadsPerGroup = MTLSizeMake(tile, tile, 1);
    MTLSize groups = MTLSizeMake(((NSUInteger)width + tile - 1) / tile, ((NSUInteger)height + tile - 1) / tile, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
