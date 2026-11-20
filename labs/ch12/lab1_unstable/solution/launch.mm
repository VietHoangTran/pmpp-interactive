// Lab 12.1 reference solution: the Metal host side, the counterparts of launchFilterAtomic,
// launchFilterWarpAggregated and launchFilterPrivatised in kernel.cu. The caller zeroes
// outputSize.
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

// One thread per key, BLOCK_DIM threads per threadgroup, or fewer if this pipeline cannot
// run that many. Fewer is safe for filter_privatised: its private list has BLOCK_DIM slots
// and a threadgroup never keeps more keys than it has threads.
// dispatchThreadgroups (not dispatchThreads) so that every threadgroup is whole: the threads
// past n exist and take part in the SIMD-group functions of filter_warp_aggregated.
static void encodeOneThreadPerKey(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> input, id<MTLBuffer> output, uint32_t n, uint32_t threshold,
                                  id<MTLBuffer> outputSize) {
    if (n == 0) return;
    NSUInteger threads = std::min<NSUInteger>(BLOCK_DIM, pipeline.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&threshold length:sizeof(threshold) atIndex:3];
    [enc setBuffer:outputSize offset:0 atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((n + threads - 1) / threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
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
