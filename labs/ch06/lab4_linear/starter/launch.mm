// Lab 6.4 starter: Metal host side (complete; your part is in kernel.metal).
#import <Metal/Metal.h>
#include <cstdint>

struct LinearDims { int32_t batch, in, out; };   // must match kernel.metal

void encodeLinear(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                  int32_t batch, int32_t in, int32_t out, int32_t tileWidth) {
    LinearDims d = {batch, in, out};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    const NSUInteger t = (NSUInteger)tileWidth;
    MTLSize groups = MTLSizeMake((out + t - 1) / t,      // x covers the outputs (columns of Y)
                                 (batch + t - 1) / t,    // y covers the batch (rows of Y)
                                 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(t, t, 1)];
}
