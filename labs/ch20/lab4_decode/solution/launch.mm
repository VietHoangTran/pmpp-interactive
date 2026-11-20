// Lab 20.4 reference solution: the Metal counterparts of launchKvAppend and
// launchDecodeAttention. Both encode into the open compute encoder they are given and
// commit nothing, so a decode step can put the append and the attention in one command
// buffer: a compute encoder runs its dispatches in order and the buffers are
// hazard-tracked, so the attention sees the appended row (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kHeadDim = 128;     // HEAD_DIM in kernel.metal
static const NSUInteger kBlockSize = 256;   // BLOCK_SIZE in kernel.metal: 8 SIMD-groups of 32

void encodeKvAppend(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> kNew, id<MTLBuffer> vNew, id<MTLBuffer> Kc, id<MTLBuffer> Vc,
                    id<MTLBuffer> pos, uint32_t batch, uint32_t maxSeq, uint32_t hKv) {
    const NSUInteger n = (NSUInteger)batch * hKv * kHeadDim;
    if (n == 0) {
        return;
    }
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:kNew offset:0 atIndex:0];
    [enc setBuffer:vNew offset:0 atIndex:1];
    [enc setBuffer:Kc offset:0 atIndex:2];
    [enc setBuffer:Vc offset:0 atIndex:3];
    [enc setBuffer:pos offset:0 atIndex:4];
    [enc setBytes:&batch length:sizeof(batch) atIndex:5];
    [enc setBytes:&maxSeq length:sizeof(maxSeq) atIndex:6];
    [enc setBytes:&hKv length:sizeof(hKv) atIndex:7];
    // One thread per element; the kernel's guard drops the surplus of the last group.
    const NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup < kBlockSize
                                    ? pipeline.maxTotalThreadsPerThreadgroup : kBlockSize;
    [enc dispatchThreadgroups:MTLSizeMake((n + perGroup - 1) / perGroup, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

void encodeDecodeAttention(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> q, id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len,
                           id<MTLBuffer> out, uint32_t batch, uint32_t hQ, uint32_t hKv,
                           uint32_t maxSeq, float scale) {
    if (batch == 0 || hQ == 0) {
        return;
    }
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:q offset:0 atIndex:0];
    [enc setBuffer:Kc offset:0 atIndex:1];
    [enc setBuffer:Vc offset:0 atIndex:2];
    [enc setBuffer:len offset:0 atIndex:3];
    [enc setBuffer:out offset:0 atIndex:4];
    [enc setBytes:&hQ length:sizeof(hQ) atIndex:5];
    [enc setBytes:&hKv length:sizeof(hKv) atIndex:6];
    [enc setBytes:&maxSeq length:sizeof(maxSeq) atIndex:7];
    [enc setBytes:&scale length:sizeof(scale) atIndex:8];
    // One threadgroup of 256 threads per (head, sequence), as dim3(h_q, batch) in kernel.cu.
    [enc dispatchThreadgroups:MTLSizeMake(hQ, batch, 1) threadsPerThreadgroup:MTLSizeMake(kBlockSize, 1, 1)];
}
