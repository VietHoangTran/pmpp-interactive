// Lab 3.2 starter: the Metal counterpart of the CUDA launch.
#import <Metal/Metal.h>
#include <cstdint>

void encodeBlur(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> in, id<MTLBuffer> out, int32_t w, int32_t h, int32_t r) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&w length:sizeof(w) atIndex:2];
    [enc setBytes:&h length:sizeof(h) atIndex:3];
    [enc setBytes:&r length:sizeof(r) atIndex:4];

    // TODO: launch with dispatchThreadgroups:threadsPerThreadgroup:, 16 by 16
    //       threads per group. This call takes the number of GROUPS, not of
    //       threads, so it is the CUDA launch in other words: how many groups
    //       cover w by h?
}
