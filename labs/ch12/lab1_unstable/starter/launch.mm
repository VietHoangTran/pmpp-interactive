// Lab 12.1 starter: the Metal host side, the counterparts of launchFilterAtomic,
// launchFilterWarpAggregated and launchFilterPrivatised in kernel.cu. The caller zeroes
// outputSize.
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

// The three kernels take the same arguments at the same indices, so one function can
// encode any of them.
static void encodeOneThreadPerKey(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> input, id<MTLBuffer> output, uint32_t n, uint32_t threshold,
                                  id<MTLBuffer> outputSize) {
    if (n == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&threshold length:sizeof(threshold) atIndex:3];
    [enc setBuffer:outputSize offset:0 atIndex:4];

    // TODO: one thread per key, BLOCK_DIM threads per threadgroup, or fewer if
    //       pipeline.maxTotalThreadsPerThreadgroup says this pipeline cannot run that many.
    //       Why is fewer safe for filter_privatised, and more not?
    //       Use dispatchThreadgroups, so that every threadgroup is whole: the threads past n
    //       must exist to take part in the SIMD-group functions.
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

void encodeFilterAtomic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> input, id<MTLBuffer> output, uint32_t n, uint32_t threshold,
                        id<MTLBuffer> outputSize) {
    encodeOneThreadPerKey(enc, pipeline, input, output, n, threshold, outputSize);
}

void encodeFilterWarpAggregated(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                id<MTLBuffer> input, id<MTLBuffer> output, uint32_t n, uint32_t threshold,
                                id<MTLBuffer> outputSize) {
    encodeOneThreadPerKey(enc, pipeline, input, output, n, threshold, outputSize);
}

void encodeFilterPrivatised(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> input, id<MTLBuffer> output, uint32_t n, uint32_t threshold,
                            id<MTLBuffer> outputSize) {
    encodeOneThreadPerKey(enc, pipeline, input, output, n, threshold, outputSize);
}
