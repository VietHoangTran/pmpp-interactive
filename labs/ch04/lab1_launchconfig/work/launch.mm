// Lab 4.1 starter: Metal host side.
// Ask the device and the pipeline, then size the threadgroup.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>

#include "launch_config.h"

struct LinearDims { uint32_t batch, in, out; };   // must match the struct in kernel.metal

// ---------- part 1: ask ----------
// Fill every field of MetalInfo (launch_config.h). Four come from the device,
// two from the pipeline state: on Metal the width of a SIMD-group and the real
// thread limit belong to a compiled kernel, not to the device.
MetalInfo queryMetal(id<MTLDevice> device, id<MTLComputePipelineState> pipeline) {
    MetalInfo m;
    std::memset(&m, 0, sizeof(m));
    // TODO 1: the property names are the field names.
    return m;
}

// ---------- part 2: derive and launch ----------
// The counterpart of launchLinear. Return the threadgroup size you dispatched with.
MTLSize encodeLinear(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                     uint32_t batch, uint32_t in, uint32_t out) {
    LinearDims d = {batch, in, out};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];

    // TODO 2: size the threadgroup from the pipeline's two properties, the way
    //         Apple's article "Calculating threadgroup and grid sizes" does,
    //         and then make it no taller than the batch.
    // TODO 3: dispatch. The kernel has no guard, so the call must be the one
    //         that takes the grid in threads, not in threadgroups.
    return MTLSizeMake(0, 0, 0);
}

// ---------- part 3: the probe's launch ----------
// The driver chooses the grid and the threadgroup size; this only dispatches.
// Buffer 0 is rec, buffer 1 is gridWidth (see kernel.metal).
void encodeProbe(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> rec, uint32_t gridWidth, uint32_t gridHeight,
                 MTLSize threadsPerThreadgroup) {
    // TODO 4: gridWidth x gridHeight threads, in threadgroups of the size given.
}
