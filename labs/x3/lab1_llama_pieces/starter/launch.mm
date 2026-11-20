// Lab X3.1 starter: the Metal counterparts of launchRmsNorm, launchRope, launchSwiGLU and
// launchRmsNormPort. Each binds its arguments into the caller's encoder, sizes any
// threadgroup memory, and dispatches. The two that return bool return false, without
// dispatching, when the pipeline cannot run the kernel as written.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;     // part A threads per threadgroup, as BLOCK_SIZE in kernel.cu
static const uint32_t kPortGrid = 4;      // part B threadgroups, as PORT_GRID in kernel.cu

// One float per SIMD-group (or per lane), rounded up to the 16-byte multiple Metal requires
// (ch05-facts C8).
static NSUInteger scratchBytes(NSUInteger count) { return (count * sizeof(float) + 15) / 16 * 16; }

// ---------------------------------------------------------------- part A
bool encodeRmsNorm(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y, uint32_t rows, uint32_t d, float eps) {
    // TODO 1: lab 10.3's check: kBlock threads in whole SIMD-groups, no more SIMD-groups than
    //         lanes, within p.maxTotalThreadsPerThreadgroup. The width is
    //         p.threadExecutionWidth: read it, never assume it.
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];            // may be x: in place
    [enc setBytes:&d length:sizeof(d) atIndex:3];
    [enc setBytes:&eps length:sizeof(eps) atIndex:4];
    // TODO 2: size scratch (one float per SIMD-group) and dispatch one threadgroup per row.
    return true;
}

void encodeRope(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                id<MTLBuffer> x, id<MTLBuffer> pos, id<MTLBuffer> cosT, id<MTLBuffer> sinT,
                uint32_t rows, uint32_t heads, uint32_t headDim, uint32_t mode) {
    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:pos offset:0 atIndex:1];
    [enc setBuffer:cosT offset:0 atIndex:2];
    [enc setBuffer:sinT offset:0 atIndex:3];
    [enc setBytes:&rows length:sizeof(rows) atIndex:4];
    [enc setBytes:&heads length:sizeof(heads) atIndex:5];
    [enc setBytes:&headDim length:sizeof(headDim) atIndex:6];
    [enc setBytes:&mode length:sizeof(mode) atIndex:7];
    // TODO 3: one thread per pair, in threadgroups of at most kBlock (and at most
    //         p.maxTotalThreadsPerThreadgroup) threads.
}

void encodeSwiGLU(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                  id<MTLBuffer> gate, id<MTLBuffer> up, id<MTLBuffer> out, uint32_t n) {
    [enc setComputePipelineState:p];
    [enc setBuffer:gate offset:0 atIndex:0];
    [enc setBuffer:up offset:0 atIndex:1];
    [enc setBuffer:out offset:0 atIndex:2];          // may be gate
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    // TODO 4: one thread per element.
}

// ---------------------------------------------------------------- part B: the port
// llama.cpp's scalar-path block size (256 under 1024 elements, else 1024), capped at what this
// pipeline runs and rounded down to whole SIMD-groups (the cap is ours). s_sum holds one slot
// per lane, because SIMD-group 0 reads one slot per lane in stage 2.
bool encodeRmsNormPort(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                       id<MTLBuffer> x, id<MTLBuffer> w, id<MTLBuffer> y, uint32_t rows, uint32_t d, float eps) {
    const NSUInteger width = p.threadExecutionWidth;
    if (width == 0) return false;
    NSUInteger threads = MIN((NSUInteger)(d < 1024 ? 256 : 1024), p.maxTotalThreadsPerThreadgroup);
    threads = threads / width * width;
    if (threads == 0 || threads / width > width) return false;
    const uint32_t grid = rows < kPortGrid ? rows : kPortGrid;
    // The float4 path pays off once the row holds at least one float4.
    const uint32_t vec = d >= 4 ? 1 : 0;

    [enc setComputePipelineState:p];
    [enc setBuffer:x offset:0 atIndex:0];
    [enc setBuffer:w offset:0 atIndex:1];
    [enc setBuffer:y offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];            // the same buffers, seen as float4
    [enc setBuffer:w offset:0 atIndex:4];
    [enc setBuffer:y offset:0 atIndex:5];
    [enc setBytes:&rows length:sizeof(rows) atIndex:6];
    [enc setBytes:&d length:sizeof(d) atIndex:7];
    [enc setBytes:&eps length:sizeof(eps) atIndex:8];
    [enc setBytes:&vec length:sizeof(vec) atIndex:9];
    [enc setThreadgroupMemoryLength:scratchBytes(width) atIndex:0];
    [enc dispatchThreadgroups:MTLSizeMake(grid, 1, 1) threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
    return true;
}
