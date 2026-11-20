// Lab 4.3: Metal host side (provided; the defects are all in the kernel).
#import <Metal/Metal.h>
#include <cstdint>

void encodeReverseBlocks(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                         id<MTLBuffer> a, NSUInteger aOffset, int32_t n, uint32_t threadsPerThreadgroup) {
    [enc setComputePipelineState:pipeline];
    // The array starts `aOffset` bytes into its buffer: the counterpart of handing
    // a CUDA kernel `pointer + k`. The driver uses it to put sentinel elements
    // before and after the array.
    [enc setBuffer:a offset:aOffset atIndex:0];
    [enc setBytes:&n length:sizeof(n) atIndex:1];

    // Whole threadgroups, like a CUDA grid: round up, and the kernel deals with a
    // short last slice. (dispatchThreads would make the last threadgroup smaller
    // instead; this lab keeps the CUDA shape so that both kernels have the same
    // three defects.)
    NSUInteger groups = (n + threadsPerThreadgroup - 1) / threadsPerThreadgroup;
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(threadsPerThreadgroup, 1, 1)];
}
