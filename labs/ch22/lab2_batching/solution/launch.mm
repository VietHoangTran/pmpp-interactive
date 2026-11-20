// Lab 22.2 reference solution: the Metal host side. Not in the book.
//
// encodeEmbed and encodeDecodeStep are the counterparts of launchEmbed and
// launchDecodeStep: they write dispatches into a compute encoder that the caller opened
// and will end, commit and wait for. The whole step is then one command buffer.
//
// Why the dispatches see each other's writes (notes/today/ch02-facts.md C9): the plain
// computeCommandEncoder "dispatches its compute commands serially", and buffers created
// from the device are hazard-tracked by default, so Metal holds a command back until an
// earlier write to a buffer it uses has finished. The step relies on both. In an encoder
// made with MTLDispatchTypeConcurrent it would need a memory barrier after every dispatch
// that the next one reads (ch22-facts C1).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>

#include "../decode.h"

void encodeEmbed(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> E, id<MTLBuffer> tokens, id<MTLBuffer> H, uint32_t batch, uint32_t d) {
    const uint32_t n = batch * d;
    if (n == 0) return;                                   // nothing to dispatch
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:E offset:0 atIndex:0];
    [enc setBuffer:tokens offset:0 atIndex:1];
    [enc setBuffer:H offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];         // a scalar: copied into the argument table
    // Exactly n threads: Metal makes the last threadgroup smaller, so the kernel needs no guard.
    NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > n) threadsPerGroup = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
}

bool encodeDecodeStep(id<MTLComputeCommandEncoder> enc, const DecodeModel& m, DecodeState& s) {
    const DecodePipelines& p = m.pipelines;
    encodeEmbed(enc, p.embed, m.E, s.tokens, s.H, s.batch, m.d);
    for (uint32_t l = 0; l < m.layers; ++l) {
        m.encodeLinear(enc, p.linear, s.H, m.W[l], s.T, s.batch, m.d, m.d);   // T = H W_l^T
        encodeAddBias(enc, p.addBias, s.T, s.T, m.b[l], s.batch, m.d);         // T += b_l
        encodeAddResidual(enc, p.addResidual, s.H, s.T, s.batch * m.d);        // H += T
    }
    m.encodeLinear(enc, p.linear, s.H, m.E, s.logits, s.batch, m.d, m.vocab);  // the LM head, tied to E
    if (!encodeSoftmaxRows(enc, p.softmax, s.logits, s.probs, s.batch, m.vocab, 1.0f / s.temperature)) {
        return false;
    }
    // The next tokens overwrite the ones this step read: the next step starts on the GPU.
    return encodeArgmaxRows(enc, p.argmax, s.logits, s.tokens, s.batch, m.vocab);
}
