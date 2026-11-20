// Lab 3.5 starter: the Metal counterpart of launchSeqMean.
// kernel.metal and this file together have the same FOUR defects as kernel.cu.
#import <Metal/Metal.h>
#include <cstdint>

void encodeSeqMean(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> in, NSUInteger inOffset, id<MTLBuffer> out, NSUInteger outOffset,
                   int32_t batch, int32_t seq, int32_t hidden, int32_t r) {
    [enc setComputePipelineState:pipeline];
    // Each tensor starts `offset` bytes into its buffer: the counterpart of
    // handing a CUDA kernel `pointer + k`. The driver uses it on a real device to
    // put guard elements before and after the tensor.
    [enc setBuffer:in  offset:inOffset  atIndex:0];
    [enc setBuffer:out offset:outOffset atIndex:1];
    [enc setBytes:&batch  length:sizeof(batch)  atIndex:2];
    [enc setBytes:&seq    length:sizeof(seq)    atIndex:3];
    [enc setBytes:&hidden length:sizeof(hidden) atIndex:4];
    [enc setBytes:&r      length:sizeof(r)      atIndex:5];

    // dispatchThreadgroups takes a number of threadgroups, like a CUDA grid: whole
    // groups only, so round up, and the kernel must turn the surplus threads away.
    MTLSize threadsPerGroup = MTLSizeMake(8, 8, 4);
    MTLSize groups = MTLSizeMake((seq    + threadsPerGroup.width  - 1) / threadsPerGroup.width,
                                 (hidden + threadsPerGroup.height - 1) / threadsPerGroup.height,
                                 (batch  + threadsPerGroup.depth  - 1) / threadsPerGroup.depth);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
