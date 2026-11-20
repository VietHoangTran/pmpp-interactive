// Lab 20.1 starter: the Metal counterpart of launchCausalSoftmax. It binds the arguments,
// sizes the per-SIMD-group scratch, and must dispatch one threadgroup of kBlock threads per
// row of every item. It returns false, without dispatching, if this pipeline cannot run the
// kernel as written.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as BLOCK_SIZE in kernel.cu

struct CausalDims { uint32_t nq, nk; float scale; };   // the same layout as in kernel.metal

// From lab 10.3's reference launch.mm (M4), unchanged. The SIMD-group width is the
// pipeline's threadExecutionWidth (ch03-facts C9), a property of the compiled pipeline. The
// kernel needs whole SIMD-groups and no more SIMD-groups than one SIMD-group has lanes
// (stage 2 gives each result one lane). Sets *groups to the number of SIMD-groups.
static bool fits(id<MTLComputePipelineState> p, NSUInteger* groups) {
    const NSUInteger width = p.threadExecutionWidth;
    if (width == 0 || kBlock > p.maxTotalThreadsPerThreadgroup || kBlock % width != 0) return false;
    *groups = kBlock / width;
    return *groups <= width;
}

// One element per SIMD-group, rounded up to the 16-byte multiple Metal requires
// (ch05-facts C8). Also from lab 10.3.
static NSUInteger scratchBytes(NSUInteger groups, NSUInteger elementSize) {
    return (groups * elementSize + 15) / 16 * 16;
}

bool encodeCausalSoftmax(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                         id<MTLBuffer> S, id<MTLBuffer> P, id<MTLBuffer> L,
                         uint32_t batch, uint32_t nq, uint32_t nk, float scale) {
    NSUInteger groups = 0;
    if (!fits(p, &groups)) return false;
    if (batch == 0 || nq == 0) return true;   // nothing to do
    CausalDims d = {nq, nk, scale};
    [enc setComputePipelineState:p];
    [enc setBuffer:S offset:0 atIndex:0];
    [enc setBuffer:P offset:0 atIndex:1];
    [enc setBuffer:L offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    [enc setThreadgroupMemoryLength:scratchBytes(groups, sizeof(float)) atIndex:0];
    // TODO: dispatch one threadgroup of kBlock threads per row of every item, rows along x
    //       and items along y:
    //   [enc dispatchThreadgroups:MTLSizeMake(?, ?, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
    return true;
}
