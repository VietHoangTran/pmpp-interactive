// Lab 4.4 starter: Metal host side.
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

    // TODO 2: a 1D dispatch covering n outputs in threadgroups of
    //         threadsPerThreadgroup threads. The tests use sizes that do not
    //         divide n. Lab 4.1 used the call that takes the grid in threads;
    //         lab 3.3 the one that takes it in threadgroups. With the first,
    //         the last threadgroup is smaller than the size being measured.
}
