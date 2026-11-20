// Lab X3.2 reference solution: the Metal host side. Not in the book.
//
// encodeLlamaStep is the counterpart of launchLlamaStep: it writes every dispatch of one
// step into the compute encoder the caller opened and will end and commit, so one step is
// one command buffer.
//
// Why each dispatch sees the previous one's writes (notes/today/ch02-facts.md C9): the
// plain computeCommandEncoder "dispatches its compute commands serially", and buffers made
// by the device are hazard-tracked by default. The step relies on both, as lab 22.2 does.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>

#include "../llama.h"

void encodeAdvancePositions(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            const LlamaState& s) {
    if (s.batch == 0) return;
    const uint32_t promptLen = s.promptLen, batch = s.batch;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:s.tokens offset:0 atIndex:0];
    [enc setBuffer:s.pos offset:0 atIndex:1];
    [enc setBuffer:s.len offset:0 atIndex:2];
    [enc setBuffer:s.prompt offset:0 atIndex:3];
    [enc setBuffer:s.out offset:0 atIndex:4];
    [enc setBuffer:s.rngStep offset:0 atIndex:5];
    [enc setBytes:&promptLen length:sizeof(promptLen) atIndex:6];
    [enc setBytes:&batch length:sizeof(batch) atIndex:7];
    // Exactly `batch` threads: Metal makes the last threadgroup smaller, so no guard is needed.
    NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (perGroup > batch) perGroup = batch;
    [enc dispatchThreads:MTLSizeMake(batch, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

bool encodeLlamaStep(id<MTLComputeCommandEncoder> enc, const LlamaModel& m, LlamaState& s) {
    const LlamaPipelines& p = m.pipelines;
    const LlamaConfig& c = m.cfg;
    const uint32_t B = s.batch, d = (uint32_t)c.dim, hidden = (uint32_t)c.hidden_dim;
    const uint32_t hq = (uint32_t)c.n_heads, hkv = (uint32_t)c.n_kv_heads, maxSeq = (uint32_t)c.seq_len;
    const uint32_t dq = hq * LLAMA_HEAD_DIM, dkv = hkv * LLAMA_HEAD_DIM, vocab = (uint32_t)c.vocab_size;
    const float scale = 1.0f / std::sqrt((float)LLAMA_HEAD_DIM);

    encodeEmbed(enc, p.embed, m.E, s.tokens, s.x, B, d);
    for (int l = 0; l < c.n_layers; ++l) {
        const LlamaLayer& w = m.layer[l];
        id<MTLBuffer> Kc = s.Kc[l], Vc = s.Vc[l];
        if (!encodeRmsNorm(enc, p.rmsNorm, s.x, w.rmsAtt, s.xb, B, d, m.eps)) return false;
        m.encodeLinear(enc, p.linear, s.xb, w.wq, s.q, B, d, dq);
        m.encodeLinear(enc, p.linear, s.xb, w.wk, s.k, B, d, dkv);
        m.encodeLinear(enc, p.linear, s.xb, w.wv, s.v, B, d, dkv);
        encodeRope(enc, p.rope, s.q, s.pos, s.cosT, s.sinT, B, hq, LLAMA_HEAD_DIM, m.ropeMode);
        encodeRope(enc, p.rope, s.k, s.pos, s.cosT, s.sinT, B, hkv, LLAMA_HEAD_DIM, m.ropeMode);
        encodeKvAppend(enc, p.kvAppend, s.k, s.v, Kc, Vc, s.pos, B, maxSeq, hkv);
        m.encodeAttention(enc, p.attention, s.q, Kc, Vc, s.len, s.att, B, hq, hkv, maxSeq, scale);
        m.encodeLinear(enc, p.linear, s.att, w.wo, s.xb2, B, dq, d);
        encodeAddResidual(enc, p.addResidual, s.x, s.xb2, B * d);
        if (!encodeRmsNorm(enc, p.rmsNorm, s.x, w.rmsFfn, s.xb, B, d, m.eps)) return false;
        m.encodeLinear(enc, p.linear, s.xb, w.w1, s.hb, B, d, hidden);
        m.encodeLinear(enc, p.linear, s.xb, w.w3, s.hb2, B, d, hidden);
        encodeSwiGLU(enc, p.swiglu, s.hb, s.hb2, s.hb, B * hidden);
        m.encodeLinear(enc, p.linear, s.hb, w.w2, s.xb2, B, hidden, d);
        encodeAddResidual(enc, p.addResidual, s.x, s.xb2, B * d);
    }
    if (!encodeRmsNorm(enc, p.rmsNorm, s.x, m.rmsFinal, s.xb, B, d, m.eps)) return false;
    m.encodeLinear(enc, p.linear, s.xb, m.E, s.logits, B, d, vocab);   // the LM head, tied to E
    if (s.encodeSample) {
        if (!s.encodeSample(enc, s.sampleCtx, s.logits, s.tokens, s.rngStep, B, vocab)) return false;
    } else if (!encodeArgmaxRows(enc, p.argmax, s.logits, s.tokens, B, vocab)) {
        return false;
    }
    encodeAdvancePositions(enc, p.advance, s);
    return true;
}
