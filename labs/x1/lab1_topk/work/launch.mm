// Lab X1.1: the Metal counterpart of launchTopKRows. Encode both kernels into the one compute
// encoder you are given, each one threadgroup of kBlock threads per row. Return false, without
// dispatching, if a pipeline cannot run its kernel as written.
//
// Ordering: a plain compute encoder dispatches serially and buffers made by the device are
// hazard-tracked, so topk_gather sees what topk_threshold wrote (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // BLOCK_SIZE in kernel.metal
static const NSUInteger kBins = 256;    // BINS in kernel.metal

bool encodeTopKRows(id<MTLComputeCommandEncoder> enc,
                    id<MTLComputePipelineState> threshold,
                    id<MTLComputePipelineState> gather,
                    id<MTLBuffer> logits, id<MTLBuffer> outIdx, id<MTLBuffer> outVal,
                    id<MTLBuffer> scratch, uint32_t rows, uint32_t cols, uint32_t k) {
    // TODO 1. Refuse (return false) unless each pipeline runs kBlock threads per threadgroup in
    //         whole SIMD-groups, with a SIMD-group width (threadExecutionWidth, a property of the
    //         compiled pipeline: ch04-facts C2) that also divides the kBins your walk covers.
    //
    // TODO 2. topk_threshold: buffers 0 and 1, cols and k as bytes at 2 and 3; then topk_gather:
    //         buffers 0 to 3, cols and k at 4 and 5. One threadgroup of kBlock threads per row.
    (void)kBins;
    return false;
}
