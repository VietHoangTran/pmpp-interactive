// Lab 3.1 reference solution: the Metal counterpart of the CUDA launch.
#import <Metal/Metal.h>
#include <cstdint>

void encodeGrayscale(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> Pin, id<MTLBuffer> Pout, uint32_t width, uint32_t height,
                     uint32_t groupW, uint32_t groupH) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:Pin offset:0 atIndex:0];
    [enc setBuffer:Pout offset:0 atIndex:1];
    // Small values need no buffer of their own: setBytes copies them into the
    // argument table, and the kernel sees them as `constant uint&`.
    [enc setBytes:&width length:sizeof(width) atIndex:2];
    [enc setBytes:&height length:sizeof(height) atIndex:3];

    // dispatchThreads takes the number of THREADS, not of threadgroups. The grid
    // is exactly the picture, x from the width and y from the height; Metal makes
    // the threadgroups on the right and bottom edges smaller when the picture is
    // not a multiple of the group shape. MTLSize has no defaults: unused is 1.
    [enc dispatchThreads:MTLSizeMake(width, height, 1)
        threadsPerThreadgroup:MTLSizeMake(groupW, groupH, 1)];
}
