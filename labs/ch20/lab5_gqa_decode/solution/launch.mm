// Lab 20.5 reference solution: the Metal counterpart of launchDecodeAttentionGrouped.
// It encodes into the open compute encoder it is given and commits nothing. The pipeline
// it receives was compiled with GQ = hQ / hKv (the driver picks it, as kernel.cu's switch
// picks a template instance).
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

static const NSUInteger kBlockSize = 256;   // BLOCK_SIZE in kernel.metal: 8 SIMD-groups of 32
static const uint32_t kMaxGq = 8;           // MAX_GQ in kernel.metal

void encodeDecodeAttentionGrouped(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                                  id<MTLBuffer> q, id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len,
                                  id<MTLBuffer> out, uint32_t batch, uint32_t hQ, uint32_t hKv,
                                  uint32_t maxSeq, float scale) {
    if (batch == 0 || hKv == 0) {
        return;
    }
    if (hQ / hKv > kMaxGq) {
        std::fprintf(stderr, "encodeDecodeAttentionGrouped: g_q = %u is above MAX_GQ = %u; nothing encoded\n",
                     hQ / hKv, kMaxGq);
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
    // One threadgroup per (K/V head, sequence), as dim3(h_kv, batch) in kernel.cu.
    [enc dispatchThreadgroups:MTLSizeMake(hKv, batch, 1) threadsPerThreadgroup:MTLSizeMake(kBlockSize, 1, 1)];
}
