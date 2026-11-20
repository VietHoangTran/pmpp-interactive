// Lab 9.3 starter: the Metal counterparts of the three CUDA launchers.
#import <Metal/Metal.h>
#include <cstdint>

// A pipeline may accept fewer threads per threadgroup than the device's maximum
// (p.maxTotalThreadsPerThreadgroup); keep your threadgroup size within it.

void encodeCountTokens(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> tokens, id<MTLBuffer> counts,
                       uint32_t batch, uint32_t len, uint32_t vocab) {
    [enc setComputePipelineState:p];
    [enc setBuffer:tokens offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBytes:&len length:sizeof(len) atIndex:2];
    [enc setBytes:&vocab length:sizeof(vocab) atIndex:3];
    // TODO: TODO 4 of kernel.cu:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

void encodeAppendTokens(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                        id<MTLBuffer> newTokens, id<MTLBuffer> counts, uint32_t batch, uint32_t vocab) {
    [enc setComputePipelineState:p];
    [enc setBuffer:newTokens offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBytes:&batch length:sizeof(batch) atIndex:2];
    [enc setBytes:&vocab length:sizeof(vocab) atIndex:3];
    // TODO: TODO 6 of kernel.cu.
}

void encodeApplyPenalties(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                          id<MTLBuffer> logits, id<MTLBuffer> counts, uint32_t batch, uint32_t vocab,
                          float frequencyPenalty, float presencePenalty) {
    const uint32_t n = batch * vocab;
    [enc setComputePipelineState:p];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&frequencyPenalty length:sizeof(frequencyPenalty) atIndex:3];
    [enc setBytes:&presencePenalty length:sizeof(presencePenalty) atIndex:4];
    // TODO: TODO 8 of kernel.cu.
}
