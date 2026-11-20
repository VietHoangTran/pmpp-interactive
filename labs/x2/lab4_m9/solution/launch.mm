// Lab X2.4 reference solution: the Metal host side. Not in the book.
//
// encodeEmbedQ, encodeQLinear and encodeQDecodeStep are the counterparts of launchEmbedQ,
// launchQLinear and launchQDecodeStep: they write dispatches into a compute encoder that the
// caller opened and will end, commit and wait for. The whole step is one command buffer.
// The plain computeCommandEncoder runs its dispatches in order and buffers made by the device
// are hazard-tracked, so each dispatch sees the previous one's writes (ch02-facts C9).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>

#include "../qdecode.h"

// Exactly n threads of `pipeline`, as lab 22.2's encodeEmbed: Metal makes the last threadgroup
// smaller, so the kernels need no guard.
static void dispatchExactly(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, uint32_t n) {
    NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (perGroup > n) perGroup = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

void encodeEmbedQ(id<MTLComputeCommandEncoder> enc, const QPipelines& p, const QMatrix& E, id<MTLBuffer> tokens,
                  id<MTLBuffer> H, uint32_t batch) {
    const uint32_t d = E.cols;
    const uint32_t n = batch * d;
    if (n == 0) return;
    switch (E.fmt) {
        case QF_F32:
            [enc setComputePipelineState:p.embedF32];
            [enc setBuffer:E.f32 offset:0 atIndex:0];
            [enc setBuffer:tokens offset:0 atIndex:1];
            [enc setBuffer:H offset:0 atIndex:2];
            [enc setBytes:&d length:sizeof(d) atIndex:3];
            dispatchExactly(enc, p.embedF32, n);
            break;
        case QF_Q8_0:
            [enc setComputePipelineState:p.embedQ8_0];
            [enc setBuffer:E.q8 offset:0 atIndex:0];
            [enc setBuffer:E.q8d offset:0 atIndex:1];
            [enc setBuffer:tokens offset:0 atIndex:2];
            [enc setBuffer:H offset:0 atIndex:3];
            [enc setBytes:&d length:sizeof(d) atIndex:4];
            dispatchExactly(enc, p.embedQ8_0, n);
            break;
        case QF_Q4_0:
            [enc setComputePipelineState:p.embedQ4_0];
            [enc setBuffer:E.q4 offset:0 atIndex:0];
            [enc setBuffer:tokens offset:0 atIndex:1];
            [enc setBuffer:H offset:0 atIndex:2];
            [enc setBytes:&d length:sizeof(d) atIndex:3];
            dispatchExactly(enc, p.embedQ4_0, n);
            break;
    }
}

void encodeQLinear(id<MTLComputeCommandEncoder> enc, const QPipelines& p, id<MTLBuffer> X, const QMatrix& W,
                   id<MTLBuffer> Y, uint32_t batch) {
    if (batch > QDECODE_MAX_BATCH) return;   // the GEMVs are matrix-vector kernels
    switch (W.fmt) {
        case QF_F32: encodeGemvGather(enc, p.gemvF32, X, W.f32, Y, batch, W.cols, W.rows); break;
        case QF_Q8_0: encodeGemvQ8_0(enc, p.gemvQ8_0, X, W.q8, W.q8d, Y, batch, W.cols, W.rows); break;
        case QF_Q4_0: encodeGemvQ4_0(enc, p.gemvQ4_0, X, W.q4, Y, batch, W.cols, W.rows); break;
    }
}

bool encodeQDecodeStep(id<MTLComputeCommandEncoder> enc, const QDecodeModel& m, DecodeState& s) {
    const QPipelines& p = m.pipelines;
    encodeEmbedQ(enc, p, m.E, s.tokens, s.H, s.batch);
    for (uint32_t l = 0; l < m.layers; ++l) {
        encodeQLinear(enc, p, s.H, m.W[l], s.T, s.batch);                    // T = H W_l^T
        encodeAddBias(enc, p.addBias, s.T, s.T, m.b[l], s.batch, m.d);        // T += b_l
        encodeAddResidual(enc, p.addResidual, s.H, s.T, s.batch * m.d);       // H += T
    }
    encodeQLinear(enc, p, s.H, m.E, s.logits, s.batch);                      // the LM head, tied to E
    if (!encodeSoftmaxRows(enc, p.softmax, s.logits, s.probs, s.batch, m.vocab, 1.0f / s.temperature)) {
        return false;
    }
    return encodeArgmaxRows(enc, p.argmax, s.logits, s.tokens, s.batch, m.vocab);
}
