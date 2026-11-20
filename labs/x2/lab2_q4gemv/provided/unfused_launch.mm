// PROVIDED, do not edit. Lab X2.2's unfused baseline on Metal: dequant_q4_0 into the scratch
// Wf, then lab 22.1's gemv_gather on Wf. Both go into the encoder given, which commits
// nothing. A plain compute encoder runs its dispatches in dispatched order and device buffers
// are hazard-tracked, so the gather sees the dequantised weights (ch02-facts C9, as in lab
// 22.1's encodeGemvSplitK).
#import <Metal/Metal.h>
#include <cstdint>

void encodeGemvGather(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> X,
                      id<MTLBuffer> W, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);   // lab 22.1

void encodeGemvQ4_0Unfused(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> dequantPipeline,
                           id<MTLComputePipelineState> gatherPipeline, id<MTLBuffer> X, id<MTLBuffer> W,
                           id<MTLBuffer> Wf, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out) {
    if (batch == 0 || out == 0 || in == 0) {
        return;
    }
    const uint32_t bytes = out * (in / 32) * 16;   // bytes of qs: one thread each
    [enc setComputePipelineState:dequantPipeline];
    [enc setBuffer:W offset:0 atIndex:0];
    [enc setBuffer:Wf offset:0 atIndex:1];
    [enc setBytes:&bytes length:sizeof(bytes) atIndex:2];
    const NSUInteger tpg = dequantPipeline.maxTotalThreadsPerThreadgroup < 256 ? dequantPipeline.maxTotalThreadsPerThreadgroup : 256;
    [enc dispatchThreadgroups:MTLSizeMake((bytes + tpg - 1) / tpg, 1, 1) threadsPerThreadgroup:MTLSizeMake(tpg, 1, 1)];

    encodeGemvGather(enc, gatherPipeline, X, Wf, Y, batch, in, out);
}
