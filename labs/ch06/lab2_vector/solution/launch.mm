// Lab 6.2 reference solution: the Metal counterpart of the two CUDA launches.
#import <Metal/Metal.h>
#include <cstdint>

void encodeVecAdd4(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n) {
    [enc setComputePipelineState:pipeline];
    // Each buffer twice: the float4 view for full chunks, the float view for the tail.
    // Offset 0 keeps the float4 view 16-byte aligned: Apple asks for a device-buffer
    // offset aligned to the element type's size (notes/today/ch06-facts.md, C3).
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:z offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:y offset:0 atIndex:4];
    [enc setBuffer:z offset:0 atIndex:5];
    [enc setBytes:&n length:sizeof(n) atIndex:6];
    NSUInteger chunks = ((NSUInteger)n + 3) / 4;   // one thread per chunk, the last may be partial
    [enc dispatchThreadgroups:MTLSizeMake((chunks + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void encodeVecAddCoarse(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:z offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    NSUInteger threads = ((NSUInteger)n + 3) / 4;  // four elements per thread
    [enc dispatchThreadgroups:MTLSizeMake((threads + 255) / 256, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
