// Lab 19.1, provided: the Metal host side of the corrected Fig. 19.11. You do not edit
// this file. The grid is the one provided/implicit_gemm.cu launches:
// (ceil(H_out*W_out / T), ceil(M / T), N) threadgroups of T x T threads.
#import <Metal/Metal.h>
#include <cstdint>

struct ImplicitGemmDims { int32_t C, H, W, M, K; };   // must match implicit_gemm.metal

void encodeConvImplicitGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            id<MTLBuffer> X, id<MTLBuffer> F, id<MTLBuffer> Y,
                            int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K,
                            int32_t tileWidth) {
    ImplicitGemmDims d = {C, H, W, M, K};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    const NSUInteger t = (NSUInteger)tileWidth;
    const NSUInteger P = (NSUInteger)(H - K + 1) * (NSUInteger)(W - K + 1);
    MTLSize groups = MTLSizeMake((P + t - 1) / t, ((NSUInteger)M + t - 1) / t, (NSUInteger)N);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(t, t, 1)];
}
