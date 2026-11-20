// Lab X1.1 reference solution: the Metal counterpart of launchTopKRows. It encodes both
// kernels into the one compute encoder it is given, each one threadgroup of kBlock threads per
// row. Returns false, without dispatching, if a pipeline cannot run its kernel as written.
//
// Ordering: a plain compute encoder dispatches serially and buffers made by the device are
// hazard-tracked, so topk_gather sees what topk_threshold wrote (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // BLOCK_SIZE in kernel.metal
static const NSUInteger kBins = 256;    // BINS in kernel.metal

// The SIMD-group width is the pipeline's threadExecutionWidth, a property of the compiled
// pipeline for which no Apple source gives a value (ch04-facts C2). Both kernels need whole
// SIMD-groups, and the bin walk needs the width to divide the 256 bins. The gather's
// per-SIMD-group arrays have kBlock entries, enough for any width.
static bool fits(id<MTLComputePipelineState> p) {
    if (p == nil || kBlock > p.maxTotalThreadsPerThreadgroup) return false;
    const NSUInteger width = p.threadExecutionWidth;
    return width != 0 && kBlock % width == 0 && kBins % width == 0;
}

bool encodeTopKRows(id<MTLComputeCommandEncoder> enc,
                    id<MTLComputePipelineState> threshold,
                    id<MTLComputePipelineState> gather,
                    id<MTLBuffer> logits, id<MTLBuffer> outIdx, id<MTLBuffer> outVal,
                    id<MTLBuffer> scratch, uint32_t rows, uint32_t cols, uint32_t k) {
    if (!fits(threshold) || !fits(gather)) return false;
    if (rows == 0 || cols == 0 || k == 0) return true;

    const MTLSize grid = MTLSizeMake(rows, 1, 1);
    const MTLSize group = MTLSizeMake(kBlock, 1, 1);

    [enc setComputePipelineState:threshold];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:scratch offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    [enc setBytes:&k length:sizeof(k) atIndex:3];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];

    [enc setComputePipelineState:gather];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:scratch offset:0 atIndex:1];
    [enc setBuffer:outIdx offset:0 atIndex:2];
    [enc setBuffer:outVal offset:0 atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
    [enc dispatchThreadgroups:grid threadsPerThreadgroup:group];
    return true;
}
