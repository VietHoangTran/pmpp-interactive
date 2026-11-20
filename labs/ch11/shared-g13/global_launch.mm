// Provided for lab 11.4, not edited: encodes the copy kernel of global_copy.metal.
// The driver checks the block size against the pipeline before it calls this.
#import <Metal/Metal.h>
#include <cstdint>

// One element per thread, `block` threads per threadgroup.
void encodeCopy(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    const uint32_t groups = (n + block - 1) / block;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}
