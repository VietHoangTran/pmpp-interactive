// Lab 9.2, provided (you do not need to edit this file): the Metal host side of the two
// comparison kernels in ref_kernels.metal. The caller zeroes bins.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

// 256 threads per threadgroup, or fewer if this pipeline cannot run 256.
static NSUInteger refThreads(id<MTLComputePipelineState> pipeline) {
    NSUInteger most = pipeline.maxTotalThreadsPerThreadgroup;
    return most < 256 ? most : 256;
}

void encodeRefHistogramBasic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> image, id<MTLBuffer> bins, uint32_t width, uint32_t height) {
    NSUInteger n = (NSUInteger)width * height, threads = refThreads(pipeline);
    if (n == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:image offset:0 atIndex:0];
    [enc setBuffer:bins offset:0 atIndex:1];
    [enc setBytes:&width length:sizeof(width) atIndex:2];
    [enc setBytes:&height length:sizeof(height) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n + threads - 1) / threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

void encodeRefHistogramContiguous(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> image, id<MTLBuffer> bins, uint32_t width, uint32_t height,
                                  uint32_t coarseFactor) {
    NSUInteger n = (NSUInteger)width * height, threads = refThreads(pipeline);
    if (n == 0 || coarseFactor == 0) return;
    NSUInteger perGroup = coarseFactor * threads;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:image offset:0 atIndex:0];
    [enc setBuffer:bins offset:0 atIndex:1];
    [enc setBytes:&width length:sizeof(width) atIndex:2];
    [enc setBytes:&height length:sizeof(height) atIndex:3];
    [enc setBytes:&coarseFactor length:sizeof(coarseFactor) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake((n + perGroup - 1) / perGroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}
