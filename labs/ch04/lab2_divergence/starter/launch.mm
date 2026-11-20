// Lab 4.2 starter: the Metal counterparts of the three CUDA launches.
#import <Metal/Metal.h>
#include <cstdint>

// Buffer indices are the ones kernel.metal declares: in 0, which or trips 1,
// out 2, n 3, and iters 4 where the kernel has it.

void encodeBranch(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> in, id<MTLBuffer> which, id<MTLBuffer> out,
                  uint32_t n, uint32_t iters, uint32_t threadsPerThreadgroup) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in    offset:0 atIndex:0];
    [enc setBuffer:which offset:0 atIndex:1];
    [enc setBuffer:out   offset:0 atIndex:2];
    [enc setBytes:&n     length:sizeof(n)     atIndex:3];
    [enc setBytes:&iters length:sizeof(iters) atIndex:4];

    // TODO 5: dispatch enough WHOLE threadgroups of threadsPerThreadgroup threads
    //   to cover n elements, in one dimension:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
    //   Use dispatchThreadgroups, not dispatchThreads. The driver's probe kernel
    //   is dispatched that way, and the specification's promise of the same cut
    //   into SIMD-groups from one threadgroup to the next leaves out the smaller
    //   trailing threadgroups that dispatchThreads creates.
}

void encodeSelect(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> in, id<MTLBuffer> which, id<MTLBuffer> out,
                  uint32_t n, uint32_t iters, uint32_t threadsPerThreadgroup) {
    // TODO 6: the same bindings and dispatch as encodeBranch.
}

void encodeLoop(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                id<MTLBuffer> in, id<MTLBuffer> trips, id<MTLBuffer> out,
                uint32_t n, uint32_t threadsPerThreadgroup) {
    // TODO 7: loop_kernel has no iters, and trips sits where which sat.
}
