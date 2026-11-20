// Lab X1.4: the Metal host side of the fused kernel, as in lab X1.2. Correct as shipped: the
// defects are in kernel.metal. Returns false, without dispatching, when the pipeline cannot run
// the kernel as written.
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>

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
