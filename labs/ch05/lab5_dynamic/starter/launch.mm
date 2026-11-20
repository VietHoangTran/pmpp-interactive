// Lab 5.5 starter: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

struct DynParams { uint32_t width, tileWidth; };   // must match kernel.metal

void encodeTiledMatmulDynamic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                              id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P,
                              uint32_t width, uint32_t tileWidth) {
    DynParams p = {width, tileWidth};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&p length:sizeof(p) atIndex:3];

    // TODO 1: tell Metal how many bytes of threadgroup memory the kernel's
    //         [[threadgroup(0)]] parameter needs (a multiple of 16):
    //   [enc setThreadgroupMemoryLength:? atIndex:0];
    // TODO 2: dispatch a grid of tileWidth x tileWidth threadgroups covering the output.
}
