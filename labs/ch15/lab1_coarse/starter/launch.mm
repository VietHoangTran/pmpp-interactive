// Lab 15.1 starter: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

// The same layout as GemmSizes in kernel.metal.
struct GemmSizes { int32_t M, K, N; };

void encodeGemmCoarse(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                      uint32_t M, uint32_t K, uint32_t N) {
    GemmSizes s = {(int32_t)M, (int32_t)K, (int32_t)N};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBytes:&s length:sizeof(s) atIndex:3];

    // TODO 10: dispatch one threadgroup per 128 x 128 tile of C (x: columns, y: rows),
    //          each a single row of 256 threads:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, ?, 1)];
}
