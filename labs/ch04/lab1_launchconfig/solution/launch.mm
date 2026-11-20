// Lab 4.1 reference solution: Metal host side.
// Ask the device and the pipeline, then size the threadgroup.
#import <Metal/Metal.h>
#include <cstdint>
#include <cstring>

#include "launch_config.h"

struct LinearDims { uint32_t batch, in, out; };   // must match the struct in kernel.metal

// ---------- part 1: ask ----------
// Metal splits the answer in two. The device knows its name and its limits for
// the simplest possible kernel. The width of a SIMD-group and the real thread
// limit belong to a compiled kernel, so they are read from the pipeline state.
// The device has no property for a core count, a clock rate, a register count
// or the number of threads resident at once.
MetalInfo queryMetal(id<MTLDevice> device, id<MTLComputePipelineState> pipeline) {
    MetalInfo m;
    std::memset(&m, 0, sizeof(m));
    std::strncpy(m.name, device.name.UTF8String, sizeof(m.name) - 1);
    MTLSize most = device.maxThreadsPerThreadgroup;
    m.maxThreadsPerThreadgroup[0] = most.width;
    m.maxThreadsPerThreadgroup[1] = most.height;
    m.maxThreadsPerThreadgroup[2] = most.depth;
    m.maxThreadgroupMemoryLength = device.maxThreadgroupMemoryLength;
    m.hasUnifiedMemory = device.hasUnifiedMemory;
    m.threadExecutionWidth = pipeline.threadExecutionWidth;
    m.maxTotalThreadsPerThreadgroup = pipeline.maxTotalThreadsPerThreadgroup;
    return m;
}

// ---------- part 2: derive and launch ----------
// The counterpart of launchLinear. Returns the threadgroup size it dispatched with.
MTLSize encodeLinear(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                     uint32_t batch, uint32_t in, uint32_t out) {
    LinearDims d = {batch, in, out};
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:W offset:0 atIndex:1];
    [enc setBuffer:Y offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];

    // Apple's recipe: one SIMD-group wide, as tall as the pipeline allows.
    NSUInteger w = pipeline.threadExecutionWidth;
    NSUInteger h = pipeline.maxTotalThreadsPerThreadgroup / w;
    // Ours: no taller than the batch. Rows beyond it would be clipped away by
    // dispatchThreads anyway; asking for them only misstates the group size.
    if (h > batch) h = batch;
    MTLSize threadsPerThreadgroup = MTLSizeMake(w, h, 1);

    // The grid is given in threads, not in threadgroups: width covers the
    // outputs, height covers the batch. Metal makes the groups at the right and
    // bottom edges smaller, so the kernel needs no guard.
    [enc dispatchThreads:MTLSizeMake(out, batch, 1) threadsPerThreadgroup:threadsPerThreadgroup];
    return threadsPerThreadgroup;
}

// ---------- part 3: the probe's launch ----------
// The driver chooses the grid and the threadgroup size; this only dispatches.
void encodeProbe(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> rec, uint32_t gridWidth, uint32_t gridHeight,
                 MTLSize threadsPerThreadgroup) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:rec offset:0 atIndex:0];
    [enc setBytes:&gridWidth length:sizeof(gridWidth) atIndex:1];
    [enc dispatchThreads:MTLSizeMake(gridWidth, gridHeight, 1) threadsPerThreadgroup:threadsPerThreadgroup];
}
