// Lab 19.1 reference solution: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <string>

struct ConvDims { int32_t C, H, W, M, K, W_grid; };   // must match kernel.metal

// Provided (provided/linear_at.mm): lab 6.4's `linear` kernel, Y = X * W^T with W stored
// (out, in), each buffer bound at a byte offset.
void encodeLinearAt(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> X, NSUInteger xOff, id<MTLBuffer> W, NSUInteger wOff,
                    id<MTLBuffer> Y, NSUInteger yOff,
                    int32_t batch, int32_t in, int32_t out, int32_t tileWidth);

// [lesson:direct]
// Fig. 19.5 in Metal: threadgroups of T x T threads, grid (M, H_grid * W_grid, N),
// both tile counts rounded up.
void encodeConvDirect(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> X, id<MTLBuffer> F, id<MTLBuffer> Y,
                      int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K, int32_t tileWidth) {
    const int32_t H_out = H - K + 1, W_out = W - K + 1;
    const int32_t W_grid = (W_out + tileWidth - 1) / tileWidth;
    const int32_t H_grid = (H_out + tileWidth - 1) / tileWidth;
    ConvDims d = {C, H, W, M, K, W_grid};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    const NSUInteger t = (NSUInteger)tileWidth;
    [enc dispatchThreadgroups:MTLSizeMake((NSUInteger)M, (NSUInteger)(H_grid * W_grid), (NSUInteger)N)
        threadsPerThreadgroup:MTLSizeMake(t, t, 1)];
}
// [/lesson:direct]

// Sample n's B^T into Bt. X is bound at sample n's first float, a byte offset: the
// Metal counterpart of passing X + n * C * H * W in CUDA.
void encodeUnroll(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> X, int32_t n, id<MTLBuffer> Bt,
                  int32_t C, int32_t H, int32_t W, int32_t K) {
    ConvDims d = {C, H, W, 0, K, 0};
    const NSUInteger P = (NSUInteger)(H - K + 1) * (NSUInteger)(W - K + 1);
    const NSUInteger total = P * (NSUInteger)(C * K * K);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:(NSUInteger)n * C * H * W * sizeof(float) atIndex:0];
    [enc setBuffer:Bt offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    const NSUInteger threads = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
    [enc dispatchThreadgroups:MTLSizeMake((total + threads - 1) / threads, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
}

// [lesson:gemm-path]
// Unroll + GEMM for every sample, all in this one compute encoder. A plain compute
// encoder runs its dispatches in order, and buffers made by the device are
// hazard-tracked, so each GEMM sees its unroll and the next unroll waits for the GEMM
// still reading Bt (notes/today/ch02-facts.md C9).
// The GEMM is lab 6.4's, called as in kernel.cu: X = F (M x C*K*K), W = Bt (P x C*K*K,
// so W^T = B), Y = sample n's M x P slice, which starts n * M * P floats into Y.
void encodeConvGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> unrollPipeline,
                    id<MTLComputePipelineState> linearPipeline,
                    id<MTLBuffer> X, id<MTLBuffer> F, id<MTLBuffer> Y, id<MTLBuffer> Bt,
                    int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K, int32_t tileWidth) {
    const int32_t P = (H - K + 1) * (W - K + 1);
    for (int32_t n = 0; n < N; ++n) {
        encodeUnroll(enc, unrollPipeline, X, n, Bt, C, H, W, K);
        encodeLinearAt(enc, linearPipeline, F, 0, Bt, 0, Y, (NSUInteger)n * M * P * sizeof(float),
                       M, C * K * K, P, tileWidth);
    }
}
// [/lesson:gemm-path]

// Provided: you do not need to change this. The scratch is one sample's B^T, P x C*K*K
// floats, and it must fit in ONE Metal buffer: Metal caps a buffer at the device's
// maxBufferLength, documented as at least 256 MB (notes/today/ch19-facts.md C2). The
// driver asks before it allocates the scratch. Empty: it fits. Otherwise: why not.
std::string convGemmScratchProblem(id<MTLDevice> device, int32_t C, int32_t H, int32_t W, int32_t K) {
    const unsigned long long P = (unsigned long long)(H - K + 1) * (unsigned long long)(W - K + 1);
    const unsigned long long bytes = P * (unsigned long long)C * K * K * sizeof(float);
    if (bytes <= device.maxBufferLength) return "";
    char b[300];
    std::snprintf(b, sizeof b,
                  "the unrolled input of one sample needs %llu bytes, more than this device's maxBufferLength "
                  "(%llu bytes): not allocated. Unroll a band of output rows at a time instead.",
                  bytes, (unsigned long long)device.maxBufferLength);
    return b;
}
