// Lab 12.4 reference solution: the Metal counterpart of launchCompactCandidates. Binds the
// arguments, sizes the per-SIMD-group totals, and dispatches one threadgroup of kBlock
// threads per row. Returns false, without dispatching, if this pipeline cannot run the
// kernel as the lab specifies it.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as BLOCK_SIZE in kernel.cu

// The SIMD-group width is the pipeline's threadExecutionWidth (notes/today/ch03-facts.md C9):
// a property of the compiled pipeline, for which no Apple source gives a value (ch04-facts C2).
// The lab asks for whole SIMD-groups and no more SIMD-groups than a SIMD-group has lanes
// (ch11-facts C2). This kernel's scan of the totals, done by every thread, would work
// without the second condition; a scan of the totals by one SIMD-group (llama.cpp's and
// PyTorch's, ch12-facts D2, D5) needs it, and the lab allows either.
// Sets *groups to the number of SIMD-groups in a threadgroup.
static bool fits(id<MTLComputePipelineState> p, NSUInteger* groups) {
    const NSUInteger width = p.threadExecutionWidth;
    if (width == 0 || kBlock > p.maxTotalThreadsPerThreadgroup || kBlock % width != 0) return false;
    *groups = kBlock / width;
    return *groups <= width;
}

bool encodeCompactCandidates(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                             id<MTLBuffer> logits, id<MTLBuffer> threshold, id<MTLBuffer> outIdx,
                             id<MTLBuffer> outVal, id<MTLBuffer> counts, uint32_t rows, uint32_t cols) {
    NSUInteger groups = 0;
    if (!fits(p, &groups)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:threshold offset:0 atIndex:1];
    [enc setBuffer:outIdx offset:0 atIndex:2];
    [enc setBuffer:outVal offset:0 atIndex:3];
    [enc setBuffer:counts offset:0 atIndex:4];
    [enc setBytes:&cols length:sizeof(cols) atIndex:5];
    // One uint per SIMD-group, rounded up to the 16-byte multiple Metal requires (ch05-facts C8).
    [enc setThreadgroupMemoryLength:(groups * sizeof(uint32_t) + 15) / 16 * 16 atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
    return true;
}
