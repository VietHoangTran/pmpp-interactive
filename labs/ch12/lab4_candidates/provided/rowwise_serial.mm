// Provided for lab 12.4 (do not edit): encodes the one-thread-per-row baseline on Metal.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kSerialGroup = 64;

void encodeCompactSerial(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> logits, id<MTLBuffer> threshold, id<MTLBuffer> outIdx,
                         id<MTLBuffer> outVal, id<MTLBuffer> counts, uint32_t rows, uint32_t cols) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:threshold offset:0 atIndex:1];
    [enc setBuffer:outIdx offset:0 atIndex:2];
    [enc setBuffer:outVal offset:0 atIndex:3];
    [enc setBuffer:counts offset:0 atIndex:4];
    [enc setBytes:&rows length:sizeof(rows) atIndex:5];
    [enc setBytes:&cols length:sizeof(cols) atIndex:6];
    const NSUInteger groups = (rows + kSerialGroup - 1) / kSerialGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kSerialGroup, 1, 1)];
}
