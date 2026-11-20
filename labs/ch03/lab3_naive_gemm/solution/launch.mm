// Lab 3.3 reference solution: Metal host side.
// The counterpart of `naiveGemmKernel<<<dimGrid, dimBlock>>>(A, B, C, j, k, l)`.
#import <Metal/Metal.h>
#include <cstdint>

struct GemmDims { uint32_t j, k, l; };   // must match the struct in kernel.metal

void encodeNaiveGemm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                     uint32_t j, uint32_t k, uint32_t l) {
    GemmDims d = {j, k, l};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];

    // MTLSize is (width, height, depth) and has no defaults: the unused depth must be 1.
    const NSUInteger side = 16;
    MTLSize threadsPerGroup = MTLSizeMake(side, side, 1);
    MTLSize groups = MTLSizeMake((l + side - 1) / side,   // width covers the columns of C
                                 (j + side - 1) / side,   // height covers the rows of C
                                 1);
    // A count of whole threadgroups, like a CUDA grid: threads past the edge of C exist,
    // and the kernel's guard stops them.
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
