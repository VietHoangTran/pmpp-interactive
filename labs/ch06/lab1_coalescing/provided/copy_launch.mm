// Provided: encodes the reference copy kernel on Metal.
#import <Metal/Metal.h>
#include <cstdint>

void encodeCopy(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> input, id<MTLBuffer> output, int32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    const NSUInteger block = 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + block - 1) / block, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}
