// Lab X1.2 reference solution: the Metal host side. encodeFilterRows composes labs 14.3, 10.3
// and 11.6 (their launch.mm files are linked into this program) with min_p_count;
// encodeFilterSortedFused dispatches the fused kernel. Each returns false, without
// dispatching, when a pipeline cannot run its kernel as written.
//
// Ordering: a plain compute encoder dispatches serially, and buffers made by the device are
// hazard-tracked, so each dispatch sees the writes of the one before
// (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>

// ---- provided by labs 14.3, 10.3 and 11.6 (their solution/launch.mm) ----
bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass, id<MTLComputePipelineState> gather,
                        id<MTLBuffer> probs, id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx,
                        id<MTLBuffer> scratch, uint32_t rows, uint32_t cols);
bool encodeSoftmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale);
bool encodeTopPMask(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> probs, id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept,
                    uint32_t rows, uint32_t cols, float p);

static const NSUInteger kBlock = 256;   // BLOCK_DIM in kernel.metal

// Whole SIMD-groups, and no more SIMD-groups than a SIMD-group has lanes, because SIMD-group 0
// gives one lane to each group's total. The width belongs to the compiled pipeline:
// threadExecutionWidth (notes/today/ch03-facts.md C9).
static bool fits(id<MTLComputePipelineState> p) {
    if (p == nil) return false;
    const NSUInteger width = p.threadExecutionWidth;
    return kBlock <= p.maxTotalThreadsPerThreadgroup && width != 0 && kBlock % width == 0 && kBlock / width <= width;
}

// Once per call, on the host, in float: llama.cpp's logf(p) (ch12-facts D3).
static float lnOf(float minP) { return minP > 0.0f ? logf(minP) : 0.0f; }

static void dispatchRows(id<MTLComputeCommandEncoder> enc, uint32_t rows) {
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
}

bool encodeMinPCount(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> sortedLogits, id<MTLBuffer> keptTopP, id<MTLBuffer> n,
                     uint32_t rows, uint32_t K, float minP) {
    if (!fits(pipeline)) return false;
    if (rows == 0 || K == 0) return true;
    const float lnMinP = lnOf(minP);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:sortedLogits offset:0 atIndex:0];
    [enc setBuffer:keptTopP offset:0 atIndex:1];
    [enc setBuffer:n offset:0 atIndex:2];
    [enc setBytes:&K length:sizeof(K) atIndex:3];
    [enc setBytes:&minP length:sizeof(minP) atIndex:4];
    [enc setBytes:&lnMinP length:sizeof(lnMinP) atIndex:5];
    dispatchRows(enc, rows);
    return true;
}

// The workspace is four buffers here, not one: lab 11.6's encodeTopPMask binds each of its
// buffers at offset 0 (README.md).
bool encodeFilterRows(id<MTLComputeCommandEncoder> enc,
                      id<MTLComputePipelineState> descKeys, id<MTLComputePipelineState> sortPass,
                      id<MTLComputePipelineState> gather, id<MTLComputePipelineState> softmax,
                      id<MTLComputePipelineState> topPMask, id<MTLComputePipelineState> minPCount,
                      id<MTLBuffer> candLogits, id<MTLBuffer> sortedPos, id<MTLBuffer> sortedLogits,
                      id<MTLBuffer> sortedProbs, id<MTLBuffer> n,
                      id<MTLBuffer> sortScratch, id<MTLBuffer> massBefore, id<MTLBuffer> keep,
                      id<MTLBuffer> keptTopP, uint32_t rows, uint32_t K, float topP, float minP) {
    if (rows == 0 || K == 0) return true;
    return encodeSortRowsDesc(enc, descKeys, sortPass, gather, candLogits, sortedLogits, sortedPos, sortScratch,
                              rows, K) &&
           encodeSoftmaxRows(enc, softmax, sortedLogits, sortedProbs, rows, K, 1.0f) &&
           encodeTopPMask(enc, topPMask, sortedProbs, massBefore, keep, keptTopP, rows, K, topP) &&
           encodeMinPCount(enc, minPCount, sortedLogits, keptTopP, n, rows, K, minP);
}

bool encodeFilterSortedFused(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> sortedLogits, id<MTLBuffer> sortedProbs, id<MTLBuffer> n,
                             uint32_t rows, uint32_t K, float topP, float minP) {
    if (!fits(pipeline)) return false;
    if (rows == 0 || K == 0) return true;
    const float lnMinP = lnOf(minP);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:sortedLogits offset:0 atIndex:0];
    [enc setBuffer:sortedProbs offset:0 atIndex:1];
    [enc setBuffer:n offset:0 atIndex:2];
    [enc setBytes:&K length:sizeof(K) atIndex:3];
    [enc setBytes:&topP length:sizeof(topP) atIndex:4];
    [enc setBytes:&minP length:sizeof(minP) atIndex:5];
    [enc setBytes:&lnMinP length:sizeof(lnMinP) atIndex:6];
    dispatchRows(enc, rows);
    return true;
}
