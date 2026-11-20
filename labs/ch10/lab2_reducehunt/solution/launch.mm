// Lab 10.2: Metal host side (provided; the defects are all in the kernel).
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

// *out = the sum of in[0 .. n-1], one threadgroup of blockDim threads per
// 2*blockDim elements.
void encodeReduceAny(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t blockDim) {
    // Stage 2 gives each lane of one SIMD-group the sum of one SIMD-group, so a
    // SIMD-group needs at least as many lanes as the threadgroup has SIMD-groups, and
    // the halving shuffle loop needs a power-of-two width. Apple documents no width;
    // threadExecutionWidth is this pipeline's SIMD-group width
    // (notes/today/ch03-facts.md C9).
    NSUInteger width = pipeline.threadExecutionWidth;
    NSUInteger simdGroups = (blockDim + width - 1) / width;
    if (blockDim > pipeline.maxTotalThreadsPerThreadgroup || simdGroups > width ||
        (width & (width - 1)) != 0) {
        std::fprintf(stderr, "encodeReduceAny: %u threads per threadgroup do not fit this pipeline "
                     "(SIMD-group width %lu, at most %lu threads)\n", blockDim, (unsigned long)width,
                     (unsigned long)pipeline.maxTotalThreadsPerThreadgroup);
        return;
    }

    // *out starts at 0, the identity of +: every threadgroup adds its sum to it. The
    // buffer has shared storage and no GPU work is using it (this command buffer is not
    // committed yet and the previous one has finished), so the CPU writes it directly.
    *(float*)out.contents = 0.0f;
    if (n == 0) return;

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    NSUInteger perGroup = 2 * (NSUInteger)blockDim;
    NSUInteger groups = (n + perGroup - 1) / perGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(blockDim, 1, 1)];
}
