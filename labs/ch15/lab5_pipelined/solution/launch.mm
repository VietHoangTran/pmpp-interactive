// Lab 15.5 reference solution: Metal host side.
#import <Metal/Metal.h>
#include <cstdint>

// The same layout as GemmSizes in kernel.metal.
struct GemmSizes { int32_t M, K, N; };

// The pipeline was compiled for one SIMD-group width (SIMD_WIDTH); the driver chooses it.
void encodeGemmPipelined(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C,
                         uint32_t M, uint32_t K, uint32_t N) {
    const uint32_t bM = 128, bN = 128, numThreads = 256;
    GemmSizes s = {(int32_t)M, (int32_t)K, (int32_t)N};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBytes:&s length:sizeof(s) atIndex:3];
    // C a second time, seen by the kernel as float4. Offset 0 keeps that view 16-byte
    // aligned: Apple asks for a buffer offset aligned to the element type's size
    // (notes/today/ch06-facts.md C3).
    [enc setBuffer:C offset:0 atIndex:4];
    // A and B a second time, as float4, for the vector loads (offset 0 for the same reason).
    [enc setBuffer:A offset:0 atIndex:5];
    [enc setBuffer:B offset:0 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((N + bN - 1) / bN, (M + bM - 1) / bM, 1)
        threadsPerThreadgroup:MTLSizeMake(numThreads, 1, 1)];
}
