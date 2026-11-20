// Lab 11.1 starter: the Metal counterparts of the CUDA launchers.
// Each returns false, without dispatching, for a block size its pipeline cannot run.
#import <Metal/Metal.h>
#include <cstdint>

static bool blockFits(id<MTLComputePipelineState> p, uint32_t block, bool simdStages) {
    // TODO 9: return false when this pipeline cannot run `block` threads:
    //   - more than p.maxTotalThreadsPerThreadgroup (the pipeline's own limit, which can
    //     be lower than the device's)
    //   - for the SIMD-group kernels (simdStages true), a block that is not a whole
    //     number of SIMD-groups of p.threadExecutionWidth threads, or one with more
    //     SIMD-groups than stage 2's single SIMD-group has lanes
    return true;
}

// One threadgroup of `block` threads per segment of `block` elements.
static void dispatchSegments(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                             id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    // TODO 2: enough threadgroups of `block` threads to cover all n elements:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
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
