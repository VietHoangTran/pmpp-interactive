// Lab 19.1 starter: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <string>

struct ConvDims { int32_t C, H, W, M, K, W_grid; };   // must match kernel.metal

// Provided (provided/linear_at.mm): lab 6.4's `linear` kernel, Y = X * W^T with W stored
// (out, in), each buffer bound at a byte offset. Offsets are in bytes.
void encodeLinearAt(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> X, NSUInteger xOff, id<MTLBuffer> W, NSUInteger wOff,
                    id<MTLBuffer> Y, NSUInteger yOff,
                    int32_t batch, int32_t in, int32_t out, int32_t tileWidth);

// Fig. 19.5 in Metal: threadgroups of tileWidth x tileWidth threads, grid (M, T, N).
void encodeConvDirect(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> X, id<MTLBuffer> F, id<MTLBuffer> Y,
                      int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K, int32_t tileWidth) {
    // TODO: W_grid and H_grid, covering every pixel (TODO 4 of kernel.cu).
    ConvDims d = {C, H, W, M, K, /* W_grid: */ 0};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    // TODO: [enc dispatchThreadgroups:MTLSizeMake(?, ?, ?) threadsPerThreadgroup:MTLSizeMake(?, ?, 1)];
}

// Sample n's B^T into Bt.
void encodeUnroll(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> X, int32_t n, id<MTLBuffer> Bt,
                  int32_t C, int32_t H, int32_t W, int32_t K) {
    ConvDims d = {C, H, W, 0, K, 0};
    [enc setComputePipelineState:pipeline];
    // TODO: bind X at index 0 so that the kernel's Xn starts at sample n. CUDA would pass
    //       the pointer X + n * C * H * W; setBuffer:offset:atIndex: takes an offset in bytes.
    [enc setBuffer:Bt offset:0 atIndex:1];
    [enc setBytes:&d length:sizeof(d) atIndex:2];
    // TODO: dispatch enough threadgroups for P * C*K*K elements. A one-dimensional group
    //       of up to pipeline.maxTotalThreadsPerThreadgroup threads is allowed.
}

// Unroll + GEMM for every sample, all in this one compute encoder: a plain compute
// encoder runs its dispatches in order, and the buffers here are hazard-tracked, so one
// scratch Bt can serve every sample (notes/today/ch02-facts.md C9).
void encodeConvGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> unrollPipeline,
                    id<MTLComputePipelineState> linearPipeline,
                    id<MTLBuffer> X, id<MTLBuffer> F, id<MTLBuffer> Y, id<MTLBuffer> Bt,
                    int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K, int32_t tileWidth) {
    // TODO: TODO 7 of kernel.cu: for each sample, encodeUnroll, then ONE encodeLinearAt.
    //       Its Y offset is where sample n's result starts, in bytes.
}

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
