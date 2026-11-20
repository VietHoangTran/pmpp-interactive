// Labs X2.3 and X2.4, provided (do not edit). STAND-INS for the Metal host side of labs X2.1
// and X2.2, used only when their solutions are absent (../shared-g12/qpieces.cmake decides,
// and builds only the encoders it needs: G12_WANT_Q8_0, G12_WANT_Q4_0). Same shape as lab
// 22.1's encodeGemvGather: one SIMD-group per output, 4 SIMD-groups per threadgroup or fewer
// if the pipeline allows fewer threads, dispatched into the caller's open encoder.
// Not in the book, and ours.
#import <Metal/Metal.h>
#include <cstdint>

static NSUInteger g12SimdGroupsPerGroup(id<MTLComputePipelineState> p) {
    const NSUInteger width = p.threadExecutionWidth;
    NSUInteger n = 4;
    while (n > 1 && n * width > p.maxTotalThreadsPerThreadgroup) --n;
    return n;
}

static void g12Dispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, uint32_t out) {
    const NSUInteger width = pipeline.threadExecutionWidth;
    const NSUInteger perGroup = g12SimdGroupsPerGroup(pipeline);
    const NSUInteger groups = (out + perGroup - 1) / perGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup * width, 1, 1)];
}

#if defined(G12_WANT_Q4_0)
// Kernel gemv_q4_0 in gemv_q4_0_g12.metal. W holds out * in / 32 blocks of 18 bytes.
void encodeGemvQ4_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> X,
                    id<MTLBuffer> W, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out) {
    if (batch == 0 || out == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(batch) atIndex:3];
    [enc setBytes:&in length:sizeof(in) atIndex:4];
    [enc setBytes:&out length:sizeof(out) atIndex:5];
    g12Dispatch(enc, pipeline, out);
}
#endif

#if defined(G12_WANT_Q8_0)
// Kernel gemv_q8_0 in gemv_q8_0_g12.metal. Wq: out * in int8 codes; Wd: out * in / 32 scales.
void encodeGemvQ8_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> X,
                    id<MTLBuffer> Wq, id<MTLBuffer> Wd, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out) {
    if (batch == 0 || out == 0) return;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:Wq offset:0 atIndex:1];
    [enc setBuffer:Wd offset:0 atIndex:2];
    [enc setBuffer:Y offset:0 atIndex:3];
    [enc setBytes:&batch length:sizeof(batch) atIndex:4];
    [enc setBytes:&in length:sizeof(in) atIndex:5];
    [enc setBytes:&out length:sizeof(out) atIndex:6];
    g12Dispatch(enc, pipeline, out);
}
#endif
