// Provided: encodes the uncoalesced baseline of lab 6.4 on Metal.
#import <Metal/Metal.h>
#include <cstdint>

struct LinearDims { int32_t batch, in, out; };   // must match linear_baseline.metal

void encodeLinearBaseline(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                          int32_t batch, int32_t in, int32_t out, int32_t tileWidth) {
    LinearDims d = {batch, in, out};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    const NSUInteger t = (NSUInteger)tileWidth;
    [enc dispatchThreadgroups:MTLSizeMake((out + t - 1) / t, (batch + t - 1) / t, 1)
        threadsPerThreadgroup:MTLSizeMake(t, t, 1)];
}
