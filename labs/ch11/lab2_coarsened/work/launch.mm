// Lab 11.2 starter: the Metal counterparts of the CUDA launchers.
// `p` was built with COARSE_FACTOR = coarse. Each encoder returns false, without
// dispatching, for a configuration its pipeline cannot run.
#import <Metal/Metal.h>
#include <cstdint>

static bool configFits(id<MTLComputePipelineState> p, uint32_t block, uint32_t coarse) {
    // TODO 5: return false when
    //   - block is more than p.maxTotalThreadsPerThreadgroup
    //   - block is not a whole number of SIMD-groups of p.threadExecutionWidth threads, or
    //     has more SIMD-groups than one SIMD-group has lanes (lab 11.1)
    //   - the threadgroup memory does not fit the device: the kernel's static arrays
    //     (p.staticThreadgroupMemoryLength) plus the segment buffer you will ask for,
    //     against p.device.maxThreadgroupMemoryLength
    return true;
}

static bool encodeSegments(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block,
                           uint32_t coarse) {
    if (!configFits(p, block, coarse)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    // TODO 6: the counterpart of the third <<< >>> argument: the length in bytes of the
    //         threadgroup buffer at [[threadgroup(0)]],
    //   [enc setThreadgroupMemoryLength:? atIndex:0];
    //         (Metal wants a multiple of 16 bytes), then enough threadgroups of `block`
    //         threads to cover all n elements, one segment of coarse*block each:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
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
