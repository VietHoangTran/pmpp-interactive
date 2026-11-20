// Lab X2.1 reference solution: the Metal counterpart of launchGemvQ8_0. It encodes into the
// open compute encoder it is given and commits nothing, so a caller can put many launches in
// one command buffer (lab X2.4 does).
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kQ8SimdGroupsPerGroup = 4;   // WARPS_PER_BLOCK in kernel.cu

// SIMD-groups per threadgroup: kQ8SimdGroupsPerGroup, or fewer if this pipeline allows fewer
// threads per threadgroup (Metal may lower the limit per pipeline: ch15-facts C3).
static NSUInteger q8SimdGroupsPerGroup(id<MTLComputePipelineState> p) {
    const NSUInteger width = p.threadExecutionWidth;
    NSUInteger n = kQ8SimdGroupsPerGroup;
    while (n > 1 && n * width > p.maxTotalThreadsPerThreadgroup) {
        --n;
    }
    return n;
}

void encodeGemvQ8_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> X,
                    id<MTLBuffer> Wq, id<MTLBuffer> Wd, id<MTLBuffer> Y, uint32_t batch, uint32_t in,
                    uint32_t out) {
    if (batch == 0 || out == 0 || in == 0) {
        return;
    }
    [enc setComputePipelineState:p];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:Wq offset:0 atIndex:1];
    [enc setBuffer:Wd offset:0 atIndex:2];
    [enc setBuffer:Y offset:0 atIndex:3];
    [enc setBytes:&batch length:sizeof(batch) atIndex:4];
    [enc setBytes:&in length:sizeof(in) atIndex:5];
    [enc setBytes:&out length:sizeof(out) atIndex:6];

    // One SIMD-group per output. The threadgroup is a whole number of SIMD-groups of the
    // width the pipeline reports (threadExecutionWidth, ch03-facts C9).
    const NSUInteger width = p.threadExecutionWidth;
    const NSUInteger perGroup = q8SimdGroupsPerGroup(p);
    const NSUInteger groups = (out + perGroup - 1) / perGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup * width, 1, 1)];
}
