// Lab X1.3: the Metal host side. Your kernels' encoders, and encodeSampleRows, which encodes
// the whole chain into the one compute encoder it is given. Each returns false, without
// dispatching, if a pipeline cannot run its kernel as written.
//
// sampler_metal.h (provided) declares the pipelines, the workspace (one buffer per
// intermediate, because the earlier labs' encoders bind at offset 0) and the earlier labs'
// encoders you compose.
//
// Ordering: a plain compute encoder dispatches serially and buffers made by the device are
// hazard-tracked, so each stage sees what the one before wrote (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

#include "sampler_metal.h"

static const NSUInteger kBlock = 256;   // BLOCK_SIZE in kernel.metal (draw_rows)

// Provided: elementwise kernels use up to 256 threads per threadgroup, or the pipeline's limit.
static NSUInteger groupSize(id<MTLComputePipelineState> p) {
    return p.maxTotalThreadsPerThreadgroup < 256 ? p.maxTotalThreadsPerThreadgroup : 256;
}

static void dispatchElementwise(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, NSUInteger count) {
    const NSUInteger g = groupSize(p);
    [enc dispatchThreadgroups:MTLSizeMake((count + g - 1) / g, 1, 1) threadsPerThreadgroup:MTLSizeMake(g, 1, 1)];
}

// Provided: copy_rows.
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

// TODO: philox_kat, one thread per vector (encodeCopyRows is the pattern).
bool encodePhiloxKat(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> in,
                     id<MTLBuffer> out, uint32_t count) {
    return false;
}

// TODO: apply_repetition over rows * vocab elements.
bool encodeApplyRepetition(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> logits,
                           id<MTLBuffer> counts, uint32_t rows, uint32_t vocab, float penalty) {
    return false;
}

// TODO: draw_rows, one threadgroup of kBlock threads per row, in whole SIMD-groups of the width
//       the pipeline reports (threadExecutionWidth, ch04-facts C2). uIn and uOut may be nil:
//       set the flags accordingly, and still bind some buffer at every index.
bool encodeDrawRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> sortedLogits,
                    id<MTLBuffer> sortedPos, id<MTLBuffer> candIdx, id<MTLBuffer> n, id<MTLBuffer> seeds,
                    id<MTLBuffer> streams, id<MTLBuffer> steps, id<MTLBuffer> uIn, id<MTLBuffer> tokens,
                    id<MTLBuffer> uOut, uint32_t rows, uint32_t stride, float invT) {
    (void)kBlock;
    return false;
}

// TODO: the chain of kernel.cu's TODO 10, with the encoders above and those sampler_metal.h
//       declares, into the buffers of ws (made for these rows, vocab and topK: check it).
bool encodeSampleRows(id<MTLComputeCommandEncoder> enc, const SamplerPipelines& p, const SampleWorkspace& ws,
                      id<MTLBuffer> logits, id<MTLBuffer> counts, id<MTLBuffer> seeds, id<MTLBuffer> streams,
                      id<MTLBuffer> steps, id<MTLBuffer> tokens, id<MTLBuffer> uOut, uint32_t rows, uint32_t vocab,
                      int topK, float topP, float minP, float temperature, float repeatPenalty,
                      float frequencyPenalty, float presencePenalty) {
    return false;
}
