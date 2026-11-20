// Provided: encodes the two in-place baseline kernels of shared/inplace_sum.metal.
// One threadgroup of `block` threads per row of 2*block floats.
#import <Metal/Metal.h>
#include <cstdint>

static void encodeInPlace(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

void encodeSimpleSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    encodeInPlace(enc, pipeline, in, out, rows, block);
}

void encodeConvergentSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    encodeInPlace(enc, pipeline, in, out, rows, block);
}
