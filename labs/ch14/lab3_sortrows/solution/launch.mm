// Lab 14.3 reference solution: the Metal counterpart of launchSortRowsDesc. It encodes the
// whole sort into the one compute encoder it is given: desc_keys, then one sort_rows_pass
// per digit, then gather_rows, each one threadgroup of kBlock threads per row. Returns false,
// without dispatching, if a pipeline cannot run the kernels as written.
//
// Ordering: a plain compute encoder dispatches serially, and buffers made by the device are
// hazard-tracked, so each dispatch sees the writes of the one before
// (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;     // BLOCK_SIZE in kernel.metal
static const uint32_t kRadixBits = 8;     // RADIX_BITS in kernel.metal: keep the two equal

// One uint per SIMD-group, rounded up to the 16-byte multiple Metal requires (ch05-facts C8).
static NSUInteger groupSumsBytes(NSUInteger groups) {
    return (groups * sizeof(uint32_t) + 15) / 16 * 16;
}

// The SIMD-group width is the pipeline's threadExecutionWidth, a property of the compiled
// pipeline that no Apple source gives a value for (ch04-facts C2). sort_rows_pass needs
// whole SIMD-groups (the scan reads each group's total from its last lane), no more
// SIMD-groups than a SIMD-group has lanes (each lane picks up one group's total), and its
// threadgroup arrays plus group_sums within the device's threadgroup memory.
// Sets *groups to the number of SIMD-groups.
static bool fits(id<MTLComputePipelineState> p, NSUInteger* groups) {
    if (p == nil || kBlock > p.maxTotalThreadsPerThreadgroup) return false;
    const NSUInteger width = p.threadExecutionWidth;
    if (width == 0 || kBlock % width != 0) return false;
    *groups = kBlock / width;
    if (*groups > width) return false;
    return p.staticThreadgroupMemoryLength + groupSumsBytes(*groups) <= p.device.maxThreadgroupMemoryLength;
}

static void dispatchRows(id<MTLComputeCommandEncoder> enc, uint32_t rows) {
    [enc dispatchThreadgroups:MTLSizeMake(rows, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
}

bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass,
                        id<MTLComputePipelineState> gather,
                        id<MTLBuffer> probs, id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx,
                        id<MTLBuffer> scratch, uint32_t rows, uint32_t cols) {
    NSUInteger groups = 0;
    if (!fits(sortPass, &groups)) return false;
    if (descKeys == nil || gather == nil || kBlock > descKeys.maxTotalThreadsPerThreadgroup ||
        kBlock > gather.maxTotalThreadsPerThreadgroup) return false;

    // scratch: keys A | indices A | keys B | indices B, rows * cols uints each. Offsets of a
    // device buffer must be aligned to the element's size (ch06-facts C3): 4 bytes here.
    const NSUInteger region = (NSUInteger)rows * cols * sizeof(uint32_t);
    const NSUInteger keysAt[2] = {0, 2 * region};
    const NSUInteger idxAt[2] = {region, 3 * region};

    [enc setComputePipelineState:descKeys];
    [enc setBuffer:probs offset:0 atIndex:0];
    [enc setBuffer:scratch offset:keysAt[0] atIndex:1];
    [enc setBuffer:scratch offset:idxAt[0] atIndex:2];
    [enc setBytes:&cols length:sizeof(cols) atIndex:3];
    dispatchRows(enc, rows);

    int cur = 0;
    for (uint32_t shift = 0; shift < 32; shift += kRadixBits) {   // least significant digit first
        [enc setComputePipelineState:sortPass];
        [enc setBuffer:scratch offset:keysAt[cur] atIndex:0];
        [enc setBuffer:scratch offset:idxAt[cur] atIndex:1];
        [enc setBuffer:scratch offset:keysAt[1 - cur] atIndex:2];
        [enc setBuffer:scratch offset:idxAt[1 - cur] atIndex:3];
        [enc setBytes:&cols length:sizeof(cols) atIndex:4];
        [enc setBytes:&shift length:sizeof(shift) atIndex:5];
        [enc setThreadgroupMemoryLength:groupSumsBytes(groups) atIndex:0];
        dispatchRows(enc, rows);
        cur = 1 - cur;
    }

    [enc setComputePipelineState:gather];
    [enc setBuffer:probs offset:0 atIndex:0];
    [enc setBuffer:scratch offset:idxAt[cur] atIndex:1];
    [enc setBuffer:sortedProbs offset:0 atIndex:2];
    [enc setBuffer:sortedIdx offset:0 atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    dispatchRows(enc, rows);
    return true;
}
