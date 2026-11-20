// Lab A.1, provided (Metal host side). Do not edit.
// One thread runs the probe of fpprobe.metal.
#import <Metal/Metal.h>

void encodeFpProbe(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> in, id<MTLBuffer> out) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:in offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake(1, 1, 1) threadsPerThreadgroup:MTLSizeMake(1, 1, 1)];
}
