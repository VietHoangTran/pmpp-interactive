// Lab 5.1 reference solution: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

struct FmaParams { uint32_t n; uint32_t iters; float a; float b; };

// ---------- provided ----------
void encodeVecAdd(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void encodeFmaChain(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t iters, float a, float b) {
    FmaParams p = {n, iters, a, b};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:2];
    [enc dispatchThreadgroups:MTLSizeMake((n + 255) / 256, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

// ---------- your part ----------
void encodeMatmul(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:M offset:0 atIndex:0];
    [enc setBuffer:N offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    const NSUInteger side = 16;
    NSUInteger groups = (width + side - 1) / side;      // round up, as in the CUDA launch
    [enc dispatchThreadgroups:MTLSizeMake(groups, groups, 1)
        threadsPerThreadgroup:MTLSizeMake(side, side, 1)];
}
