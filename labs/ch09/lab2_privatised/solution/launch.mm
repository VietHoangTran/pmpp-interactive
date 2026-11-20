// Lab 9.2: the Metal host side, the counterpart of launchHistogramShared and
// launchHistogramRuns in kernel.cu. The caller zeroes bins.
#import <Metal/Metal.h>
#include <cstdint>

// 256 threads per threadgroup (or fewer if this pipeline cannot run 256), each taking
// coarseFactor pixels: ceil(n / (coarseFactor*threads)) threadgroups. Fewer threadgroups
// than one per 256 pixels is the point of coarsening: each one pays once to zero and once
// to commit its private bins (section 9.5).
static void encodeCoarsened(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> image, id<MTLBuffer> bins, uint32_t width, uint32_t height,
                            uint32_t coarseFactor) {
    NSUInteger n = (NSUInteger)width * height;
    if (n == 0 || coarseFactor == 0) return;
    NSUInteger most = pipeline.maxTotalThreadsPerThreadgroup;
    NSUInteger threads = most < 256 ? most : 256;
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

void encodeHistogramShared(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> image, id<MTLBuffer> bins, uint32_t width, uint32_t height,
                           uint32_t coarseFactor) {
    encodeCoarsened(enc, pipeline, image, bins, width, height, coarseFactor);
}

void encodeHistogramRuns(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> image, id<MTLBuffer> bins, uint32_t width, uint32_t height,
                         uint32_t coarseFactor) {
    encodeCoarsened(enc, pipeline, image, bins, width, height, coarseFactor);
}
