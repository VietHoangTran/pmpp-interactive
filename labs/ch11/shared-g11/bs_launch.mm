// Provided for labs 11.1 and 11.2, not edited: encodes the provided Metal kernels of
// this directory. The lab drivers check the block size against the pipeline before they
// call these, so nothing here refuses a block.
#import <Metal/Metal.h>
#include <cstdint>

static void bindAndDispatch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                            id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n,
                            uint32_t groups, uint32_t block) {
    [enc setComputePipelineState:p];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

// Fig. 11.3, one threadgroup per segment of `block` elements.
void encodeKoggeStone(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    bindAndDispatch(enc, p, in, out, n, (n + block - 1) / block, block);
}

// One element per thread.
void encodeCopy(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block) {
    bindAndDispatch(enc, p, in, out, n, (n + block - 1) / block, block);
}

// The thread-major scan; `p` was built with COARSE_FACTOR = coarse.
void encodeThreadMajorScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block,
                           uint32_t coarse) {
    const uint32_t segment = coarse * block;
    bindAndDispatch(enc, p, in, out, n, (n + segment - 1) / segment, block);
}
