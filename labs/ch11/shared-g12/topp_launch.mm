// Provided for lab 11.6 (do not edit): encodes the one-thread-per-row baseline
// (topp_naive.metal) and the copy (topp_copy.metal) on Metal. The driver compiles their
// pipelines from this directory and passes them in.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kNaiveGroup = 64;   // threads per threadgroup, one row each
static const NSUInteger kCopyGroup = 256;   // threads per threadgroup, one element each

void encodeTopPMaskNaive(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> probs, id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept,
                         uint32_t rows, uint32_t cols, float p) {
    if (rows == 0 || cols == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:probs offset:0 atIndex:0];
    [enc setBuffer:massBefore offset:0 atIndex:1];
    [enc setBuffer:keep offset:0 atIndex:2];
    [enc setBuffer:kept offset:0 atIndex:3];
    [enc setBytes:&rows length:sizeof(rows) atIndex:4];
    [enc setBytes:&cols length:sizeof(cols) atIndex:5];
    [enc setBytes:&p length:sizeof(p) atIndex:6];
    const NSUInteger groups = (rows + kNaiveGroup - 1) / kNaiveGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kNaiveGroup, 1, 1)];
}

void encodeTopPCopy(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> in,
                    id<MTLBuffer> outF, id<MTLBuffer> outB, uint32_t n) {
    if (n == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:outF offset:0 atIndex:1];
    [enc setBuffer:outB offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    const NSUInteger groups = (n + kCopyGroup - 1) / kCopyGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kCopyGroup, 1, 1)];
}
