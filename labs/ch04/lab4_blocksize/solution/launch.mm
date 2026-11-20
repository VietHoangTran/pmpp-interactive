// Lab 4.4 reference solution: Metal host side.
// The counterpart of launchGatherSum: the threadgroup size is an argument.
#import <Metal/Metal.h>
#include <cstdint>

struct GatherDims { uint32_t n, k; };   // must match the struct in kernel.metal

void encodeGatherSum(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> table, id<MTLBuffer> idx, id<MTLBuffer> out,
                     uint32_t n, uint32_t k, uint32_t threadsPerThreadgroup) {
    GatherDims d = {n, k};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:table offset:0 atIndex:0];
    [enc setBuffer:idx offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];

    // Whole threadgroups, as a CUDA grid is whole blocks: the benchmark is about
    // the threadgroup size, so no threadgroup should be smaller than the others.
    // The kernel has the guard that this needs.
    NSUInteger groups = (n + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
}
