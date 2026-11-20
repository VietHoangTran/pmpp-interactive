// Lab 3.2 reference solution: the Metal counterpart of the CUDA launch.
#import <Metal/Metal.h>
#include <cstdint>

void encodeBlur(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> in, id<MTLBuffer> out, int32_t w, int32_t h, int32_t r) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    [enc setBytes:&h length:sizeof(h) atIndex:3];
    [enc setBytes:&r length:sizeof(r) atIndex:4];

    // dispatchThreadgroups takes the number of GROUPS, like a CUDA grid. Whole
    // groups only, so round up: the grid can be larger than the picture and the
    // kernel must test its position.
    const NSUInteger side = 16;
    MTLSize groups = MTLSizeMake((w + side - 1) / side, (h + side - 1) / side, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(side, side, 1)];
}
