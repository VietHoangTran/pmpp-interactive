// Lab X1.3 reference solution: the Metal host side. Your four kernels' encoders, and
// encodeSampleRows, which encodes the whole chain into the one compute encoder it is given.
// Each returns false, without dispatching, if a pipeline cannot run its kernel as written.
//
// Ordering: a plain compute encoder dispatches serially and buffers made by the device are
// hazard-tracked, so each stage sees what the one before wrote (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

#include "sampler_metal.h"

static const NSUInteger kBlock = 256;   // BLOCK_SIZE in kernel.metal (draw_rows)

// Elementwise kernels: up to 256 threads per threadgroup, or the pipeline's own limit.
static NSUInteger groupSize(id<MTLComputePipelineState> p) {
    return p.maxTotalThreadsPerThreadgroup < 256 ? p.maxTotalThreadsPerThreadgroup : 256;
}

static void dispatchElementwise(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger count) {
    const NSUInteger g = groupSize(p);
    [enc dispatchThreadgroups:MTLSizeMake((count + g - 1) / g, 1, 1) threadsPerThreadgroup:MTLSizeMake(g, 1, 1)];
}

bool encodeCopyRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> in,
                    id<MTLBuffer> out, uint32_t count) {
    if (p == nil) return false;
    if (count == 0) return true;
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    dispatchElementwise(enc, p, count);
    return true;
}

bool encodePhiloxKat(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> in,
                     id<MTLBuffer> out, uint32_t count) {
    if (p == nil) return false;
    if (count == 0) return true;
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:2];
    dispatchElementwise(enc, p, count);
    return true;
}

bool encodeApplyRepetition(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> logits,
                           id<MTLBuffer> counts, uint32_t rows, uint32_t vocab, float penalty) {
    if (p == nil) return false;
    const uint32_t n = rows * vocab;   // the tests and benchmark stay far below 2^32 elements
    if (n == 0) return true;
    [enc setComputePipelineState:p];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&penalty length:sizeof(penalty) atIndex:3];
    dispatchElementwise(enc, p, n);
    return true;
}

bool encodeDrawRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> sortedLogits,
                    id<MTLBuffer> sortedPos, id<MTLBuffer> candIdx, id<MTLBuffer> n, id<MTLBuffer> seeds,
                    id<MTLBuffer> streams, id<MTLBuffer> steps, id<MTLBuffer> uIn, id<MTLBuffer> tokens,
                    id<MTLBuffer> uOut, uint32_t rows, uint32_t stride, float invT) {
    // Whole SIMD-groups of a width the pipeline reports (ch04-facts C2); the kernel's
    // per-SIMD-group arrays have kBlock entries, enough for any width.
    if (p == nil || kBlock > p.maxTotalThreadsPerThreadgroup) return false;
    const NSUInteger width = p.threadExecutionWidth;
    if (width == 0 || kBlock % width != 0) return false;
    if (rows == 0) return true;
    const uint32_t flags = (uIn != nil ? 1u : 0u) | (uOut != nil ? 2u : 0u);
    // A buffer the kernel does not read still gets a binding: `tokens` stands in.
    [enc setComputePipelineState:p];
    [enc setBuffer:sortedLogits offset:0 atIndex:0];
    [enc setBuffer:sortedPos offset:0 atIndex:1];
    [enc setBuffer:candIdx offset:0 atIndex:2];
    [enc setBuffer:n offset:0 atIndex:3];
    [enc setBuffer:(seeds ? seeds : tokens) offset:0 atIndex:4];
    [enc setBuffer:(streams ? streams : tokens) offset:0 atIndex:5];
    [enc setBuffer:(steps ? steps : tokens) offset:0 atIndex:6];
    [enc setBuffer:(uIn ? uIn : tokens) offset:0 atIndex:7];
    [enc setBuffer:tokens offset:0 atIndex:8];
    [enc setBuffer:(uOut ? uOut : tokens) offset:0 atIndex:9];
    [enc setBytes:&stride length:sizeof(stride) atIndex:10];
    [enc setBytes:&invT length:sizeof(invT) atIndex:11];
    [enc setBytes:&flags length:sizeof(flags) atIndex:12];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
    return true;
}

bool encodeSampleRows(id<MTLComputeCommandEncoder> enc, const SamplerPipelines& p, const SampleWorkspace& ws,
                      id<MTLBuffer> logits, id<MTLBuffer> counts, id<MTLBuffer> seeds, id<MTLBuffer> streams,
                      id<MTLBuffer> steps, id<MTLBuffer> tokens, id<MTLBuffer> uOut, uint32_t rows, uint32_t vocab,
                      int topK, float topP, float minP, float temperature, float repeatPenalty,
                      float frequencyPenalty, float presencePenalty) {
    const uint32_t K = effectiveTopK(vocab, topK);
    if (ws.rows != rows || ws.vocab != vocab || ws.k != K) return false;
    if (rows == 0 || vocab == 0) return true;
    if (p.applyPenalties == nil) return false;

    // Penalties on a copy; a neutral setting is skipped, as llama.cpp skips it.
    bool ok = encodeCopyRows(enc, p.copyRows, logits, ws.logits, rows * vocab);
    if (ok && repeatPenalty != 1.0f) ok = encodeApplyRepetition(enc, p.applyRepetition, ws.logits, counts, rows, vocab, repeatPenalty);
    if (ok && (frequencyPenalty != 0.0f || presencePenalty != 0.0f))
        encodeApplyPenalties(enc, p.applyPenalties, ws.logits, counts, rows, vocab, frequencyPenalty, presencePenalty);
    if (!ok) return false;

    if (temperature <= 0.0f) {   // greedy: lab 10.3's argmax, lowest id on ties; uOut not written
        return encodeArgmaxRows(enc, p.argmax, ws.logits, tokens, rows, vocab);
    }
    return encodeTopKRows(enc, p.topkThreshold, p.topkGather, ws.logits, ws.candIdx, ws.candVal, ws.selScratch,
                          rows, vocab, K) &&
           encodeSortRowsDesc(enc, p.descKeys, p.sortPass, p.sortGather, ws.candVal, ws.sortedVal, ws.sortedPos,
                              ws.sortScratch, rows, K) &&
           encodeSoftmaxRows(enc, p.softmax, ws.sortedVal, ws.probs, rows, K, 1.0f) &&
           encodeTopPMask(enc, p.topP, ws.probs, ws.mass, ws.keep, ws.kept, rows, K, topP) &&
           encodeMinPCombine(enc, p.minP, ws.sortedVal, ws.kept, ws.n, rows, K, minP) &&
           encodeDrawRows(enc, p.drawRows, ws.sortedVal, ws.sortedPos, ws.candIdx, ws.n, seeds, streams, steps, nil,
                          tokens, uOut, rows, K, 1.0f / temperature);
}
