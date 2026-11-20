// Lab 4.2 reference solution: the Metal counterparts of the three CUDA launches.
#import <Metal/Metal.h>
#include <cstdint>

// Whole threadgroups, like a CUDA grid: round up, and the kernel turns the
// surplus threads away. The driver's probe kernel is dispatched in exactly this
// shape. The MSL specification promises the same cut into SIMD-groups from one
// threadgroup to the next within a dispatch, except for the smaller trailing
// threadgroups that dispatchThreads creates (section 4.4.1).
static void dispatchWhole(id<MTLComputeCommandEncoder> enc, uint32_t n, uint32_t threadsPerThreadgroup) {
    NSUInteger groups = (n + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
}

void encodeBranch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> in, id<MTLBuffer> which, id<MTLBuffer> out,
                  uint32_t n, uint32_t iters, uint32_t threadsPerThreadgroup) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in    offset:0 atIndex:0];
    [enc setBuffer:which offset:0 atIndex:1];
    [enc setBuffer:out   offset:0 atIndex:2];
    [enc setBytes:&n     length:sizeof(n)     atIndex:3];
    [enc setBytes:&iters length:sizeof(iters) atIndex:4];
    dispatchWhole(enc, n, threadsPerThreadgroup);
}

void encodeSelect(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> in, id<MTLBuffer> which, id<MTLBuffer> out,
                  uint32_t n, uint32_t iters, uint32_t threadsPerThreadgroup) {
    // Same bindings, a different pipeline.
    encodeBranch(enc, pipeline, in, which, out, n, iters, threadsPerThreadgroup);
}

void encodeLoop(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> in, id<MTLBuffer> trips, id<MTLBuffer> out,
                uint32_t n, uint32_t threadsPerThreadgroup) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in    offset:0 atIndex:0];
    [enc setBuffer:trips offset:0 atIndex:1];
    [enc setBuffer:out   offset:0 atIndex:2];
    [enc setBytes:&n     length:sizeof(n) atIndex:3];
    dispatchWhole(enc, n, threadsPerThreadgroup);
}
