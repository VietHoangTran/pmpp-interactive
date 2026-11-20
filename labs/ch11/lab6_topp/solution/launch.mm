// Lab 11.6 reference solution: the Metal host side of the top-p keep mask.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, BLOCK_DIM in kernel.metal

// One threadgroup of kBlock threads per row. Returns false, without dispatching, when this
// pipeline cannot run the kernel as written.
bool encodeTopPMask(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> probs, id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept,
                    uint32_t rows, uint32_t cols, float p) {
    // Whole SIMD-groups, and no more SIMD-groups than a SIMD-group has lanes, because
    // SIMD-group 0 gives one lane to each SIMD-group's total. The width belongs to the
    // compiled pipeline: threadExecutionWidth (notes/today/ch03-facts.md C9).
    const NSUInteger width = pipeline.threadExecutionWidth;
    if (kBlock > pipeline.maxTotalThreadsPerThreadgroup || width == 0 || kBlock % width != 0 ||
        kBlock / width > width) {
        return false;
    }
    if (rows == 0 || cols == 0) return true;
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:probs offset:0 atIndex:0];
    [enc setBuffer:massBefore offset:0 atIndex:1];
    [enc setBuffer:keep offset:0 atIndex:2];
    [enc setBuffer:kept offset:0 atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
    return true;
}
