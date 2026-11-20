// Lab 11.2 reference solution: the Metal counterparts of the CUDA launchers.
// `p` was built with COARSE_FACTOR = coarse. Each encoder returns false, without
// dispatching, for a configuration its pipeline cannot run.
#import <Metal/Metal.h>
#include <cstdint>

// The block must fit the pipeline (ch05-facts C3), be whole SIMD-groups with at most
// `width` of them (the block scan's stage 2 runs in one SIMD-group), and the threadgroup
// memory, the kernel's static arrays plus the segment buffer, must fit the device
// (ch05-facts C3, C8).
static bool configFits(id<MTLComputePipelineState> p, uint32_t block, uint32_t coarse) {
    if (block == 0 || coarse == 0 || block > p.maxTotalThreadsPerThreadgroup) return false;
    const NSUInteger width = p.threadExecutionWidth;
    if (block % width != 0 || block / width > width) return false;
    const NSUInteger segmentBytes = (NSUInteger)coarse * block * sizeof(float);
    return p.staticThreadgroupMemoryLength + segmentBytes <= p.device.maxThreadgroupMemoryLength;
}

static bool encodeSegments(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block,
                           uint32_t coarse) {
    if (!configFits(p, block, coarse)) return false;
    const uint32_t segment = coarse * block;
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    // The counterpart of the third <<< >>> argument. The length must be a multiple of
    // 16 bytes (ch05-facts C8); with block >= 64 it always is.
    [enc setThreadgroupMemoryLength:segment * sizeof(float) atIndex:0];
    const uint32_t groups = (n + segment - 1) / segment;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
    return true;
}

bool encodeCoarsenedScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                         id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block,
                         uint32_t coarse) {
    return encodeSegments(enc, p, in, out, n, block, coarse);
}

bool encodeRegisterTiledScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                             id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block,
                             uint32_t coarse) {
    return encodeSegments(enc, p, in, out, n, block, coarse);
}
