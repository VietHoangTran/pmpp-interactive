// Lab 10.3 reference solution: the Metal counterparts of launchRowMax, launchSoftmaxRows
// and launchArgmaxRows. Each binds its arguments, sizes the per-SIMD-group scratch, and
// dispatches one threadgroup of kBlock threads per row. Each returns false, without
// dispatching, if this pipeline cannot run the kernel as written.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as BLOCK_SIZE in kernel.cu

// The SIMD-group width is the pipeline's threadExecutionWidth (notes/today/ch03-facts.md C9):
// a property of the compiled pipeline, not of the device, for which no Apple source gives a
// value (ch04-facts C2). The kernels need whole SIMD-groups and no more SIMD-groups than one
// SIMD-group has lanes (stage 2 gives each result one lane).
// Sets *groups to the number of SIMD-groups in a threadgroup.
static bool fits(id<MTLComputePipelineState> p, NSUInteger* groups) {
    const NSUInteger width = p.threadExecutionWidth;
    if (width == 0 || kBlock > p.maxTotalThreadsPerThreadgroup || kBlock % width != 0) return false;
    *groups = kBlock / width;
    return *groups <= width;
}

// One element per SIMD-group, rounded up to the 16-byte multiple Metal requires
// (ch05-facts C8).
static NSUInteger scratchBytes(NSUInteger groups, NSUInteger elementSize) {
    return (groups * elementSize + 15) / 16 * 16;
}

static void dispatchRows(id<MTLComputeCommandEncoder> enc, uint32_t rows) {
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
}

bool encodeRowMax(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                  id<MTLBuffer> x, id<MTLBuffer> out, uint32_t rows, uint32_t cols) {
    NSUInteger groups = 0;
    if (!fits(p, &groups)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:out offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    [enc setThreadgroupMemoryLength:scratchBytes(groups, sizeof(float)) atIndex:0];
    dispatchRows(enc, rows);
    return true;
}

bool encodeSoftmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> x, id<MTLBuffer> y, uint32_t rows, uint32_t cols, float scale) {
    NSUInteger groups = 0;
    if (!fits(p, &groups)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:y offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    [enc setBytes:&scale length:sizeof(scale) atIndex:3];
    [enc setThreadgroupMemoryLength:scratchBytes(groups, sizeof(float)) atIndex:0];
    dispatchRows(enc, rows);
    return true;
}

bool encodeArgmaxRows(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                      id<MTLBuffer> x, id<MTLBuffer> idx, uint32_t rows, uint32_t cols) {
    NSUInteger groups = 0;
    if (!fits(p, &groups)) return false;
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:idx offset:0 atIndex:1];
    [enc setBytes:&cols length:sizeof(cols) atIndex:2];
    [enc setThreadgroupMemoryLength:scratchBytes(groups, sizeof(float)) atIndex:0];     // values
    [enc setThreadgroupMemoryLength:scratchBytes(groups, sizeof(int32_t)) atIndex:1];   // indices
    dispatchRows(enc, rows);
    return true;
}
