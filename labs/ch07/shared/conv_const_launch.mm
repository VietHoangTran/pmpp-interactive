// Lab 7.2, provided: dispatches the Metal baseline (conv_const.metal). Do not edit.
#import <Metal/Metal.h>
#include <cstdint>

void encodeConvConstBaseline(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                             int32_t width, int32_t height) {
    const NSUInteger block = 16;   // as in conv_const.cu
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:N offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc setBytes:&height length:sizeof(height) atIndex:4];
    MTLSize groups = MTLSizeMake(((NSUInteger)width + block - 1) / block, ((NSUInteger)height + block - 1) / block, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(block, block, 1)];
}
