// Lab 10.1 starter: the Metal counterparts of the three CUDA launchers.
// Each returns false, without dispatching, for a block size this pipeline cannot run.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>

static bool blockFits(id<MTLComputePipelineState> p, uint32_t block, bool simdStages) {
    // TODO 8: return false when this pipeline cannot run `block` threads:
    //   - more than p.maxTotalThreadsPerThreadgroup (the pipeline's own limit, which can
    //     be lower than the device's)
    //   - for the SIMD-group kernels (simdStages true), a block that is not a whole
    //     number of SIMD-groups of p.threadExecutionWidth threads, or one with more
    //     SIMD-groups than stage 2's single SIMD-group has lanes
    return true;
}

bool encodeSharedMemorySum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    if (!blockFits(p, block, false)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    // TODO 2: one threadgroup of `block` threads per row:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
    return true;
}

bool encodeTwoStageWarpSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    if (!blockFits(p, block, true)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    // TODO 5: as TODO 2.
    return true;
}

bool encodeCoarsenedSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block, uint32_t coarse) {
    if (!blockFits(p, block, true) || coarse == 0) return false;
    // TODO 7a: *out must hold 0 before the threadgroups add into it, on every call.
    //          A compute encoder has no cudaMemset. This buffer is in shared storage
    //          (out.contents is a CPU pointer to it), and the command buffer that will
    //          run this dispatch has not been committed yet.
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&coarse length:sizeof(coarse) atIndex:3];
    // TODO 7b: enough threadgroups of `block` threads to cover all n elements.
    return true;
}
