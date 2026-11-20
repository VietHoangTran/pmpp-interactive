// Lab 7.1 starter: the Metal counterparts of launchConvBasic, launchConvConst and
// launchConvByValue. Dispatch TILE_WIDTH x TILE_WIDTH threadgroups (tileWidth here),
// rounded up to cover the image, as the CUDA launches do; the kernels guard the
// overhang. The buffer indices must match the [[buffer(n)]] attributes in kernel.metal.
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

    // TODO 1: dispatch enough tileWidth x tileWidth threadgroups to cover width x height:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, ?, 1)];
}

// @part const
// F holds the weights already (the driver filled it). conv_const declares it in the
// constant address space.
void encodeConvConst(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> N, id<MTLBuffer> F, id<MTLBuffer> P,
                     int32_t r, int32_t width, int32_t height, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    // TODO 2: bind N, the filter, P, r, width and height to the indices conv_const
    //         expects. Does binding a buffer for the constant address space need a
    //         different call from binding one for the device address space?
    // TODO 3: dispatch as in encodeConvBasic.
}

// @part byvalue
// Provided. It must match struct Filter in kernel.metal byte for byte.
struct Filter {
    int32_t r;
    float w[MAX_FILTER_DIM * MAX_FILTER_DIM];
};
static_assert(sizeof(Filter) == 904, "Filter must be 4 + 225 * 4 bytes, as in kernel.metal");

// F_h is the filter in HOST memory: (2r+1) * (2r+1) floats.
void encodeConvByValue(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> N, const float* F_h, id<MTLBuffer> P,
                       int32_t r, int32_t width, int32_t height, uint32_t tileWidth) {
    [enc setComputePipelineState:pipeline];
    // TODO 4: fill a Filter from r and F_h, and hand it to the kernel at buffer(1) with
    //         setBytes, so that it travels with this dispatch the way a CUDA argument
    //         passed by value travels with a launch. No MTLBuffer is needed.
    // TODO 5: bind N, P, width and height to the indices conv_by_value expects, and
    //         dispatch as in encodeConvBasic.
}
