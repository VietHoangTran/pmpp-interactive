// Lab 3.5 reference solution: the Metal counterpart of launchSeqMean.
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
    // FIX D: MTLSize is (width, height, depth), lowest dimension first; the tensor
    // is written [batch, seq, hidden], highest first. Width must cover hidden and
    // height must cover seq. The starter sized width from seq and height from hidden.
    MTLSize groups = MTLSizeMake((hidden + threadsPerGroup.width  - 1) / threadsPerGroup.width,
                                 (seq    + threadsPerGroup.height - 1) / threadsPerGroup.height,
                                 (batch  + threadsPerGroup.depth  - 1) / threadsPerGroup.depth);
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threadsPerGroup];
}
