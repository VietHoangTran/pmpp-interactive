// Lab 20.5 starter: the Metal counterpart of launchDecodeAttentionGrouped. It encodes into
// the open compute encoder it is given and commits nothing. The pipeline it receives was
// compiled with GQ = hQ / hKv (the driver picks it, as kernel.cu's switch picks a template
// instance).
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

static const NSUInteger kBlockSize = 256;   // BLOCK_SIZE in kernel.metal
static const uint32_t kMaxGq = 8;           // MAX_GQ in kernel.metal

void encodeDecodeAttentionGrouped(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
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
    // TODO 6: refuse hQ / hKv above kMaxGq (print a message, encode nothing), then one
    //         threadgroup of kBlockSize threads per (K/V head, sequence):
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}
