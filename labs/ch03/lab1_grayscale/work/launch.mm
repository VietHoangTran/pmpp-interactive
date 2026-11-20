// Lab 3.1 starter: the Metal counterpart of the CUDA launch.
#import <Metal/Metal.h>
#include <cstdint>

// groupW x groupH is the threadgroup shape, Metal's name for the block shape.
void encodeGrayscale(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> Pin, id<MTLBuffer> Pout, uint32_t width, uint32_t height,
                     uint32_t groupW, uint32_t groupH) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:Pin offset:0 atIndex:0];
    [enc setBuffer:Pout offset:0 atIndex:1];

    // TODO 1: hand `width` and `height` to the kernel's buffer slots 2 and 3.
    //         They are too small to deserve an MTLBuffer each: use the encoder's
    //         setBytes:length:atIndex:.

    // TODO 2: launch with dispatchThreads:threadsPerThreadgroup:, which takes the
    //         size of the whole grid in threads, so the grid can be exactly the
    //         picture. Both arguments are MTLSizeMake(?, ?, 1).
}
