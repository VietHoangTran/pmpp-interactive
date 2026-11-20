// Lab 11.1 reference solution: the Metal counterparts of the CUDA launchers.
// Each returns false, without dispatching, for a block size its pipeline cannot run.
#import <Metal/Metal.h>
#include <cstdint>

// The block must fit the pipeline (its own limit, which may be below the device's:
// ch05-facts C3). The SIMD-group kernels also need whole SIMD-groups (the width divides
// the block) and at most `width` of them, so that stage 2's single SIMD-group has one
// lane per SIMD-group total.
static bool blockFits(id<MTLComputePipelineState> p, uint32_t block, bool simdStages) {
    if (block == 0 || block > p.maxTotalThreadsPerThreadgroup) return false;
    if (!simdStages) return true;
    const NSUInteger width = p.threadExecutionWidth;
    return block % width == 0 && block / width <= width;
}

// One threadgroup of `block` threads per segment of `block` elements.
static void dispatchSegments(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                             id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    const uint32_t groups = (n + block - 1) / block;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

bool encodeKoggeStoneDB(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    if (!blockFits(p, block, false)) return false;
    dispatchSegments(enc, p, in, out, n, block);
    return true;
}

bool encodeWarpScanShuffle(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    if (!blockFits(p, block, true)) return false;
    dispatchSegments(enc, p, in, out, n, block);
    return true;
}

bool encodeWarpScanPrefix(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                          id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    if (!blockFits(p, block, true)) return false;
    dispatchSegments(enc, p, in, out, n, block);
    return true;
}
