// Lab 6.2 starter: the Metal counterpart of the two CUDA launches.
#import <Metal/Metal.h>
#include <cstdint>

void encodeVecAdd4(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n) {
    [enc setComputePipelineState:pipeline];
    // TODO 1: bind x, y and z at the indices vec_add4 reads them as float4, and again at
    //         the indices it reads them as float, then n. Apple asks for a device-buffer
    //         offset aligned to the element type's size (notes/today/ch06-facts.md, C3).
    // TODO 2: one thread per chunk of four (count the partial last chunk), 256 per threadgroup:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}

void encodeVecAddCoarse(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBuffer:z offset:0 atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    // TODO 3: enough threads for four elements each, 256 per threadgroup:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
}
