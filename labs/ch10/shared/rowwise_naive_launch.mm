// Provided for lab 10.3 (do not edit): encodes the one-thread-per-row baseline on Metal.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kNaiveGroup = 64;

static void dispatchRows(id<MTLComputeCommandEncoder> enc, uint32_t rows) {
    const NSUInteger groups = (rows + kNaiveGroup - 1) / kNaiveGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(kNaiveGroup, 1, 1)];
}

void encodeSoftmaxRowsNaive(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    [enc setBytes:&scale length:sizeof(scale) atIndex:4];
    dispatchRows(enc, rows);
}

void encodeArgmaxRowsNaive(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> x, id<MTLBuffer> idx, uint32_t rows, uint32_t cols) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:idx offset:0 atIndex:1];
    [enc setBytes:&rows length:sizeof(rows) atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    dispatchRows(enc, rows);
}
