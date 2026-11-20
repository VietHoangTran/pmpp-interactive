// Lab 5.3 starter: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

struct GemmDims { uint32_t j, k, l; };   // must match the struct in kernel.metal

void encodeGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                uint32_t j, uint32_t k, uint32_t l, uint32_t tileWidth) {
    GemmDims d = {j, k, l};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    // TODO: dispatch a grid of threadgroups that covers the j x l output.
}
