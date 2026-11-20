// Lab 15.3 starter: Metal host side (lab 15.2's, renamed).
#import <Metal/Metal.h>
#include <cstdint>

// The same layout as GemmSizes in kernel.metal.
struct GemmSizes { int32_t M, K, N; };

void encodeGemmRemap(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                     uint32_t M, uint32_t K, uint32_t N) {
    const uint32_t bM = 128, bN = 128, numThreads = 256;
    GemmSizes s = {(int32_t)M, (int32_t)K, (int32_t)N};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBytes:&s length:sizeof(s) atIndex:3];
    // TODO 3: bind C a second time, at the index where kernel.metal declares its float4
    //         view. Which buffer offset keeps that view 16-byte aligned?
    // One threadgroup per 128 x 128 tile of C (x: columns, y: rows), 256 threads in a row.
    [enc dispatchThreadgroups:MTLSizeMake((N + bN - 1) / bN, (M + bM - 1) / bM, 1)
        threadsPerThreadgroup:MTLSizeMake(numThreads, 1, 1)];
}
