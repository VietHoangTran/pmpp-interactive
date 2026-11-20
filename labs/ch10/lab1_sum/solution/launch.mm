// Lab 10.1 reference solution: the Metal counterparts of the three CUDA launchers.
// Each returns false, without dispatching, for a block size this pipeline cannot run.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>

// The block must fit the pipeline (its own limit, which may be below the device's:
// notes/today/ch05-facts.md C3). The SIMD-group kernels also need whole SIMD-groups
// (width divides the block) and at most `width` of them, so that stage 2's single
// SIMD-group has one lane per partial sum.
static bool blockFits(id<MTLComputePipelineState> p, uint32_t block, bool simdStages) {
    if (block == 0 || block > p.maxTotalThreadsPerThreadgroup) return false;
    if (!simdStages) return true;
    const NSUInteger width = p.threadExecutionWidth;
    return block % width == 0 && block / width <= width;
}

static void bindRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                     id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    // One threadgroup per row of 2*block floats.
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

bool encodeSharedMemorySum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    if (!blockFits(p, block, false)) return false;
    bindRows(enc, p, in, out, rows, block);
    return true;
}

bool encodeTwoStageWarpSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block) {
    if (!blockFits(p, block, true)) return false;
    bindRows(enc, p, in, out, rows, block);
    return true;
}

bool encodeCoarsenedSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block, uint32_t coarse) {
    if (!blockFits(p, block, true) || coarse == 0) return false;
    // The counterpart of cudaMemset(out, 0, 4). The buffer is in shared storage and the
    // command buffer has not been committed yet, so the CPU writes it now and the
    // dispatch below sees 0 (ch05-facts C6: the CPU and GPU must not use it at once).
    std::memset(out.contents, 0, sizeof(float));
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&coarse length:sizeof(coarse) atIndex:3];
    const uint32_t perGroup = 2 * coarse * block;
    const uint32_t groups = (n + perGroup - 1) / perGroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    return true;
}
