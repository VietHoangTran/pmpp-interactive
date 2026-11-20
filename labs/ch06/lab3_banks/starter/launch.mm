// Lab 6.3 starter: the Metal counterpart of launchTiledTranspose.
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

    // TODO: one tileWidth x tileWidth threadgroup per tile of the input, enough of them
    //       to cover a matrix that is not a multiple of the tile:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, ?, 1)];
}
