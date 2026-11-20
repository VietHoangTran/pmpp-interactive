// Provided: encodes lab 6.3's two instruments on Metal, the bank probe
// (provided/bank_probe.metal) and the copy (provided/copy.metal). Do not edit.
#import <Metal/Metal.h>
#include <cstdint>

// One threadgroup of `lanes` threads; the driver passes the pipeline's threadExecutionWidth.
void encodeBankProbe(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> out, uint32_t lanes, int32_t stride, int32_t iters, float zero) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:out offset:0 atIndex:0];
    [enc setBytes:&stride length:sizeof(stride) atIndex:1];
    [enc setBytes:&iters length:sizeof(iters) atIndex:2];
    [enc setBytes:&zero length:sizeof(zero) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(lanes, 1, 1)];
}

void encodeCopy(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> in, id<MTLBuffer> out, int32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    NSUInteger groups = ((NSUInteger)n + 255) / 256;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
