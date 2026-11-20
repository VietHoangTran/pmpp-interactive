// Lab 20.4 starter: the Metal counterparts of launchKvAppend and launchDecodeAttention.
// Both encode into the open compute encoder they are given and commit nothing, so the
// driver can put a decode step's append and attention in one command buffer: a compute
// encoder runs its dispatches in order and the buffers are hazard-tracked.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kHeadDim = 128;     // HEAD_DIM in kernel.metal
static const NSUInteger kBlockSize = 256;   // BLOCK_SIZE in kernel.metal

void encodeKvAppend(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> kNew, id<MTLBuffer> vNew, id<MTLBuffer> Kc, id<MTLBuffer> Vc,
                    id<MTLBuffer> pos, uint32_t batch, uint32_t maxSeq, uint32_t hKv) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:kNew offset:0 atIndex:0];
    [enc setBuffer:vNew offset:0 atIndex:1];
    [enc setBuffer:Kc offset:0 atIndex:2];
    [enc setBuffer:Vc offset:0 atIndex:3];
    [enc setBuffer:pos offset:0 atIndex:4];
    [enc setBytes:&batch length:sizeof(batch) atIndex:5];
    [enc setBytes:&maxSeq length:sizeof(maxSeq) atIndex:6];
    [enc setBytes:&hKv length:sizeof(hKv) atIndex:7];
    // TODO: one thread per element of k_new (batch * hKv * kHeadDim of them), in
    //       threadgroups of at most kBlockSize threads (and at most the pipeline's
    //       maxTotalThreadsPerThreadgroup):
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

void encodeDecodeAttention(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                           id<MTLBuffer> q, id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len,
                           id<MTLBuffer> out, uint32_t batch, uint32_t hQ, uint32_t hKv,
                           uint32_t maxSeq, float scale) {
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
    // TODO: one threadgroup of kBlockSize threads per (head, sequence), as dim3(h_q, batch)
    //       in kernel.cu:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}
