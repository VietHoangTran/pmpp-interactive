// Lab 7.1 reference solution: the Metal counterparts of launchConvBasic,
// launchConvConst and launchConvByValue. Each dispatches TILE_WIDTH x TILE_WIDTH
// threadgroups, rounded up to cover the image, as the CUDA launches do; the
// kernels guard the overhang.
#import <Metal/Metal.h>
#include <cstdint>

#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// @part basic
void encodeConvBasic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                     int32_t r, int32_t width, int32_t height, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:N offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&r length:sizeof(r) atIndex:3];
    [enc setBytes:&width length:sizeof(width) atIndex:4];
    [enc setBytes:&height length:sizeof(height) atIndex:5];

    MTLSize threadsPerGroup = MTLSizeMake(tileWidth, tileWidth, 1);
    MTLSize groups = MTLSizeMake((width + tileWidth - 1) / tileWidth, (height + tileWidth - 1) / tileWidth, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}

// @part const
// The host side is the same call as for conv_basic: a buffer is bound with setBuffer
// whatever address space the kernel declares for it. Here the kernel takes it as
// `constant float*`. There is no symbol to copy to; the driver filled F with the
// weights, as it did for conv_basic.
void encodeConvConst(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                     int32_t r, int32_t width, int32_t height, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:N offset:0 atIndex:0];
    [enc setBuffer:F offset:0 atIndex:1];   // conv_const reads it as constant float*
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&r length:sizeof(r) atIndex:3];
    [enc setBytes:&width length:sizeof(width) atIndex:4];
    [enc setBytes:&height length:sizeof(height) atIndex:5];

    MTLSize threadsPerGroup = MTLSizeMake(tileWidth, tileWidth, 1);
    MTLSize groups = MTLSizeMake((width + tileWidth - 1) / tileWidth, (height + tileWidth - 1) / tileWidth, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}

// @part byvalue
// Must match struct Filter in kernel.metal byte for byte.
struct Filter {
    int32_t r;
    float w[MAX_FILTER_DIM * MAX_FILTER_DIM];
};
static_assert(sizeof(Filter) == 904, "Filter must be 4 + 225 * 4 bytes, as in kernel.metal");

// F_h is the filter in host memory. setBytes copies the struct into the command
// (Apple's limit for setBytes is 4 KB), so each dispatch carries its own weights,
// like a CUDA kernel argument passed by value.
void encodeConvByValue(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> N, const float* F_h, id<MTLBuffer> P,
                       int32_t r, int32_t width, int32_t height, uint32_t tileWidth) {
    Filter f = {};   // zero the unused tail of w
    f.r = r;
    for (int i = 0; i < (2 * r + 1) * (2 * r + 1); i++) f.w[i] = F_h[i];

    [enc setComputePipelineState:pipeline];
    [enc setBuffer:N offset:0 atIndex:0];
    [enc setBytes:&f length:sizeof(f) atIndex:1];
    [enc setBuffer:P offset:0 atIndex:2];
    [enc setBytes:&width length:sizeof(width) atIndex:3];
    [enc setBytes:&height length:sizeof(height) atIndex:4];

    MTLSize threadsPerGroup = MTLSizeMake(tileWidth, tileWidth, 1);
    MTLSize groups = MTLSizeMake((width + tileWidth - 1) / tileWidth, (height + tileWidth - 1) / tileWidth, 1);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
