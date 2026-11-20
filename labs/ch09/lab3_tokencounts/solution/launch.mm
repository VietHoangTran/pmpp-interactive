// Lab 9.3 reference solution: the Metal counterparts of the three CUDA launchers.
#import <Metal/Metal.h>
#include <cstdint>

// 256 threads per threadgroup, or fewer if this pipeline cannot run that many
// (its own limit may be below the device's: notes/today/ch05-facts.md C3).
static NSUInteger groupSize(id<MTLComputePipelineState> p) {
    return p.maxTotalThreadsPerThreadgroup < 256 ? p.maxTotalThreadsPerThreadgroup : 256;
}

void encodeCountTokens(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> tokens, id<MTLBuffer> counts,
                       uint32_t batch, uint32_t len, uint32_t vocab) {
    [enc setComputePipelineState:p];
    [enc setBuffer:tokens offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBytes:&len length:sizeof(len) atIndex:2];
    [enc setBytes:&vocab length:sizeof(vocab) atIndex:3];
    // Grid: x = the threadgroups of one row, y = the row (request).
    const NSUInteger group = groupSize(p);
    NSUInteger groupsPerRow = (len + group - 1) / group;
    if (groupsPerRow > 1024) groupsPerRow = 1024;
    [enc dispatchThreadgroups:MTLSizeMake(groupsPerRow, batch, 1)
        threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}

void encodeAppendTokens(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                        id<MTLBuffer> newTokens, id<MTLBuffer> counts, uint32_t batch, uint32_t vocab) {
    [enc setComputePipelineState:p];
    [enc setBuffer:newTokens offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBytes:&batch length:sizeof(batch) atIndex:2];
    [enc setBytes:&vocab length:sizeof(vocab) atIndex:3];
    const NSUInteger group = groupSize(p);
    [enc dispatchThreadgroups:MTLSizeMake((batch + group - 1) / group, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
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
    const NSUInteger group = groupSize(p);
    [enc dispatchThreadgroups:MTLSizeMake((n + group - 1) / group, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}
