// Lab 22.1 reference solution: the Metal counterparts of launchGemvGather and
// launchGemvSplitK. Both encode into the open compute encoder they are given and commit
// nothing, so a caller can put many launches in one command buffer.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kSimdGroupsPerGroup = 4;   // WARPS_PER_BLOCK in kernel.cu

// SIMD-groups per threadgroup: kSimdGroupsPerGroup, or fewer if this pipeline allows fewer
// threads per threadgroup (Metal may lower the limit per pipeline: ch15-facts C3).
static NSUInteger simdGroupsPerGroup(id<MTLComputePipelineState> p) {
    const NSUInteger width = p.threadExecutionWidth;
    NSUInteger n = kSimdGroupsPerGroup;
    while (n > 1 && n * width > p.maxTotalThreadsPerThreadgroup) {
        --n;
    }
    return n;
}

static void bindProblem(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                        uint32_t batch, uint32_t in, uint32_t out) {
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(batch) atIndex:3];
    [enc setBytes:&in length:sizeof(in) atIndex:4];
    [enc setBytes:&out length:sizeof(out) atIndex:5];
}

// ---------------------------------------------------------------- (a) gather
void encodeGemvGather(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                      uint32_t batch, uint32_t in, uint32_t out) {
    if (batch == 0 || out == 0) {
        return;
    }
    [enc setComputePipelineState:pipeline];
    bindProblem(enc, X, W, Y, batch, in, out);

    // One SIMD-group per output. The threadgroup is a whole number of SIMD-groups, with
    // the width the pipeline reports ("The number of threads in a SIMD group is returned
    // by the threadExecutionWidth of your compute pipeline state object", ch03-facts C9).
    const NSUInteger width = pipeline.threadExecutionWidth;
    const NSUInteger perGroup = simdGroupsPerGroup(pipeline);
    const NSUInteger groups = (out + perGroup - 1) / perGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup * width, 1, 1)];
}

// ---------------------------------------------------------------- (b) split-K, provided
void encodeGemvSplitK(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> splitPipeline,
                      id<MTLComputePipelineState> zeroPipeline, id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                      uint32_t batch, uint32_t in, uint32_t out, uint32_t splits) {
    if (batch == 0 || out == 0 || splits == 0) {
        return;
    }
    // 1. Y = 0, the counterpart of cudaMemset. It is a dispatch, not a CPU write, so that
    //    several launches can share one command buffer. The encoder is serial ("Command
    //    encoder dispatches are executed in dispatched order") and the buffer is
    //    hazard-tracked, so gemv_splitk below sees the zeros (ch02-facts C9).
    const uint32_t n = batch * out;
    [enc setComputePipelineState:zeroPipeline];
    [enc setBuffer:Y offset:0 atIndex:0];
    [enc setBytes:&n length:sizeof(n) atIndex:1];
    const NSUInteger zt = zeroPipeline.maxTotalThreadsPerThreadgroup < 256 ? zeroPipeline.maxTotalThreadsPerThreadgroup : 256;
    [enc dispatchThreadgroups:MTLSizeMake((n + zt - 1) / zt, 1, 1) threadsPerThreadgroup:MTLSizeMake(zt, 1, 1)];

    // 2. One SIMD-group per (chunk, output) pair.
    [enc setComputePipelineState:splitPipeline];
    bindProblem(enc, X, W, Y, batch, in, out);
    [enc setBytes:&splits length:sizeof(splits) atIndex:6];
    const NSUInteger width = splitPipeline.threadExecutionWidth;
    const NSUInteger perGroup = simdGroupsPerGroup(splitPipeline);
    const NSUInteger simdGroups = (NSUInteger)splits * out;
    const NSUInteger groups = (simdGroups + perGroup - 1) / perGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup * width, 1, 1)];
}
