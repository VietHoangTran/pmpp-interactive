// Lab 3.3 starter: Metal host side.
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

    // TODO: dispatch 16 x 16 threadgroups, enough of them to cover C for any j and l.
    //       MTLSize is (width, height, depth) and has no defaults: unused sizes must be 1.
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, ?, 1)];
}
