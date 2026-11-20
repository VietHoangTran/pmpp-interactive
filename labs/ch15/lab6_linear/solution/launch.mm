// Lab 15.6 reference solution: Metal host side of milestone M3.
// Unlike M2's encodeLinear there is no tile argument: the tile is fixed at 128 x 128, and
// the SIMD-group width (SIMD_WIDTH) is compiled into the pipeline the caller passes.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>

struct LinearDims { int32_t batch, in, out; };   // the same layout as in kernel.metal

void encodeLinear(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                  uint32_t batch, uint32_t in, uint32_t out) {
    const uint32_t bM = 128, bN = 128, numThreads = 256;
    // Metal may give a pipeline a lower thread limit than the device's (ch15-facts C3).
    // A threadgroup larger than the pipeline allows is not dispatched.
    if (pipeline.maxTotalThreadsPerThreadgroup < numThreads) {
        std::fprintf(stderr, "encodeLinear: this pipeline allows %lu threads per threadgroup, the kernel needs %u\n",
                     (unsigned long)pipeline.maxTotalThreadsPerThreadgroup, numThreads);
        return;
    }
    LinearDims d = {(int32_t)batch, (int32_t)in, (int32_t)out};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    // Y, X and W a second time, seen by the kernel as float4. Offset 0 keeps those views
    // 16-byte aligned: Apple asks for a buffer offset aligned to the element type's size
    // (notes/today/ch06-facts.md C3).
    [enc setBuffer:Y offset:0 atIndex:4];
    [enc setBuffer:X offset:0 atIndex:5];
    [enc setBuffer:W offset:0 atIndex:6];
    [enc dispatchThreadgroups:MTLSizeMake((out + bN - 1) / bN,      // x covers the outputs (columns of Y)
                                          (batch + bM - 1) / bM,    // y covers the batch (rows of Y)
                                          1)
        threadsPerThreadgroup:MTLSizeMake(numThreads, 1, 1)];
}
