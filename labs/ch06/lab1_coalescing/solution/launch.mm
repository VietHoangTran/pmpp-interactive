// Lab 6.1 reference solution: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

struct StridedArgs { int32_t n, stride; };      // must match kernel.metal
struct TransposeDims { int32_t rows, cols; };   // must match kernel.metal

void encodeStridedRead(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> input, id<MTLBuffer> output, int32_t n, int32_t stride) {
    StridedArgs a = {n, stride};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&a length:sizeof(a) atIndex:2];
    const NSUInteger block = 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + block - 1) / block, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

static void encodeTranspose(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> input, id<MTLBuffer> output, int32_t rows, int32_t cols,
                            NSUInteger gridX, NSUInteger gridY) {
    TransposeDims d = {rows, cols};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    const NSUInteger side = 16;
    [enc dispatchThreadgroups:MTLSizeMake((gridX + side - 1) / side, (gridY + side - 1) / side, 1)
        threadsPerThreadgroup:MTLSizeMake(side, side, 1)];
}

void encodeTransposeReadRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> input, id<MTLBuffer> output, int32_t rows, int32_t cols) {
    encodeTranspose(enc, pipeline, input, output, rows, cols, cols, rows);   // x covers the columns of in
}

void encodeTransposeWriteRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                              id<MTLBuffer> input, id<MTLBuffer> output, int32_t rows, int32_t cols) {
    encodeTranspose(enc, pipeline, input, output, rows, cols, rows, cols);   // x covers the columns of out
}
