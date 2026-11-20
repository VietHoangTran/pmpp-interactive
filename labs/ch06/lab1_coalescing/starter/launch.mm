// Lab 6.1 starter: Metal host side.
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
    // TODO: dispatch 1D threadgroups that cover n threads (round up):
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

void encodeTransposeReadRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> input, id<MTLBuffer> output, int32_t rows, int32_t cols) {
    TransposeDims d = {rows, cols};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    // TODO: 16 x 16 threadgroups, enough of them to cover the matrix this kernel's x and y walk.
}

void encodeTransposeWriteRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                              id<MTLBuffer> input, id<MTLBuffer> output, int32_t rows, int32_t cols) {
    TransposeDims d = {rows, cols};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    // TODO: 16 x 16 threadgroups, enough of them to cover the matrix this kernel's x and y walk.
}
