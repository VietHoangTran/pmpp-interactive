// Provided: encodes the one-element-per-thread baseline on Metal. Do not edit.
#import <Metal/Metal.h>
#include <cstdint>

void encodeVecAdd(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:z offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger groups = ((NSUInteger)n + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
