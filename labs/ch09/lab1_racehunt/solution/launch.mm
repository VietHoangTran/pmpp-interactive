// Lab 9.1: Metal host side (provided and correct; the defects are all in the kernels).
// The Metal counterparts of launchHistogramBasic and launchHistogramPool in kernel.cu.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

// One threadgroup of threadsPerBlock threads per threadsPerBlock pixels. Returns the
// number of threadgroups, or 0 if nothing can be dispatched.
static NSUInteger threadgroupsFor(id<MTLComputePipelineState> pipeline, uint32_t width, uint32_t height,
                                  uint32_t threadsPerBlock) {
    if (threadsPerBlock == 0 || threadsPerBlock > pipeline.maxTotalThreadsPerThreadgroup) {
        std::fprintf(stderr, "histogram: %u threads per threadgroup do not fit this pipeline (at most %lu)\n",
                     threadsPerBlock, (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
        return 0;
    }
    NSUInteger n = (NSUInteger)width * height;
    return (n + threadsPerBlock - 1) / threadsPerBlock;
}

// The caller zeroes bins.
void encodeHistogramBasic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                          id<MTLBuffer> image, id<MTLBuffer> bins,
                          uint32_t width, uint32_t height, uint32_t threadsPerBlock) {
    NSUInteger groups = threadgroupsFor(pipeline, width, height, threadsPerBlock);
    if (groups == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:image offset:0 atIndex:0];
    [enc setBuffer:bins offset:0 atIndex:1];
    [enc setBytes:&width length:sizeof(width) atIndex:2];
    [enc setBytes:&height length:sizeof(height) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerBlock, 1, 1)];
}

// The caller zeroes bins and the pool, which holds one private histogram of 256
// counters per threadgroup.
void encodeHistogramPool(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> image, id<MTLBuffer> bins, id<MTLBuffer> binsPool,
                         uint32_t width, uint32_t height, uint32_t threadsPerBlock) {
    NSUInteger groups = threadgroupsFor(pipeline, width, height, threadsPerBlock);
    if (groups == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:image offset:0 atIndex:0];
    [enc setBuffer:bins offset:0 atIndex:1];
    [enc setBuffer:binsPool offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc setBytes:&height length:sizeof(height) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerBlock, 1, 1)];
}
