// Lab 5.1 starter: Metal host side.
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
// The Metal counterpart of `matmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width)`.
void encodeMatmul(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width) {
    [enc setComputePipelineState:pipeline];
    // TODO 1: bind M, N, P to buffer indices 0, 1, 2 and `width` to index 3
    //         (see encodeVecAdd above for the calls).
    // TODO 2: dispatch enough 2D threadgroups to cover a width x width output
    //         for any width. "threadgroup" is Metal's word for a CUDA block.
}
