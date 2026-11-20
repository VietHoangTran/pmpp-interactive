// Lab X2.3: the Metal host side (provided; the defects are all in the GEMV kernel).
// Both functions encode into the open compute encoder they are given and commit nothing, so
// the quantiser and the GEMV can share one command buffer. The plain compute encoder runs its
// dispatches in order and buffers made by the device are hazard-tracked, so the GEMV sees the
// quantiser's output (ch02-facts C9). Not in the book.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kSimdGroupsPerRow = 4;   // WARPS in kernel.cu and kernel.metal

// Kernel quantize_q8_1: one thread per value. Returns false, without dispatching, if the
// pipeline's SIMD-group width is not a multiple of 32: the kernel's 32-value groups would
// then straddle two SIMD-groups. Apple documents no fixed width (infra-facts C2, C5).
bool encodeQuantizeQ8_1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> X,
                        id<MTLBuffer> Xq, uint32_t batch, uint32_t in) {
    const NSUInteger width = pipeline.threadExecutionWidth;
    if (width == 0 || width % 32 != 0 || in % 32 != 0) return false;
    const uint32_t n = batch * in;
    if (n == 0) return true;
    NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup < 256 ? pipeline.maxTotalThreadsPerThreadgroup : 256;
    perGroup -= perGroup % width;                  // whole SIMD-groups
    if (perGroup == 0) return false;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:Xq offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((n + perGroup - 1) / perGroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
    return true;
}

// Kernel gemv_q4_0_q8_1: one threadgroup per output row, kSimdGroupsPerRow SIMD-groups of the
// pipeline's width, or fewer if the pipeline allows fewer threads (ch15-facts C3).
void encodeGemvQ4_0Q8_1(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, id<MTLBuffer> W,
                        id<MTLBuffer> Xq, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out) {
    if (batch == 0 || out == 0) return;
    const NSUInteger width = pipeline.threadExecutionWidth;
    NSUInteger groups = kSimdGroupsPerRow;
    while (groups > 1 && groups * width > pipeline.maxTotalThreadsPerThreadgroup) --groups;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:W offset:0 atIndex:0];
    [enc setBuffer:Xq offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&batch length:sizeof(batch) atIndex:3];
    [enc setBytes:&in length:sizeof(in) atIndex:4];
    [enc setBytes:&out length:sizeof(out) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(out, 1, 1) threadsPerThreadgroup:MTLSizeMake(groups * width, 1, 1)];
}
