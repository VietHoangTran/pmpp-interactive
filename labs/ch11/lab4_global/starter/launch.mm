// Lab 11.4 starter: the Metal counterparts of the CUDA launchers.
//
// A whole global scan is several dispatches written into one compute encoder that the
// caller opened and will close, commit and wait for. Each dispatch must see what the
// one before it wrote. Apple documents that it does (ch02-facts C9): the plain
// computeCommandEncoder "dispatches its compute commands serially", and buffers created
// from the device are hazard-tracked by default. No threadgroup may wait for another one
// inside a dispatch: Metal documents no forward progress between threadgroups
// (ch11-facts C5), and the hierarchical scan does not need any.
//
// Each encode function returns false, without dispatching anything, for a block size
// one of its pipelines cannot run.
#import <Metal/Metal.h>
#include <cstdint>

// One definition for Metal: the driver builds kernel.metal with this value.
#define COARSE_FACTOR 4
int coarseFactor() { return COARSE_FACTOR; }

static uint32_t segmentsOf(uint32_t n, uint32_t block) {
    const uint32_t segment = COARSE_FACTOR * block;
    return (n + segment - 1) / segment;
}

// How many floats of scratch the encoders below need for n elements.
uint32_t scratchFloats(uint32_t n, uint32_t block) {
    // TODO 6: as in kernel.cu.
    return 0;
}

// Provided: whether pipeline p can run threadgroups of `block` threads. The block must
// fit the pipeline (ch05-facts C3) and be whole SIMD-groups, at most `width` of them
// (stage 2 of the block scan and of the reduction runs in one SIMD-group). With a
// segment buffer, the threadgroup memory must fit the device (ch05-facts C3, C8).
static bool fits(id<MTLComputePipelineState> p, uint32_t block, bool segmentBuffer) {
    if (p == nil || block == 0 || block > p.maxTotalThreadsPerThreadgroup) return false;
    const NSUInteger width = p.threadExecutionWidth;
    if (block % width != 0 || block / width > width) return false;
    const NSUInteger extra = segmentBuffer ? (NSUInteger)COARSE_FACTOR * block * sizeof(float) : 0;
    return p.staticThreadgroupMemoryLength + extra <= p.device.maxThreadgroupMemoryLength;
}

// The levels of the recursion live inside one scratch buffer. setBuffer:offset:atIndex:
// binds a buffer starting at a byte offset; for a float array the offset must be a
// multiple of 4 (ch06-facts C3). A scan kernel also needs
// setThreadgroupMemoryLength:COARSE_FACTOR*block*sizeof(float) atIndex:0, the
// counterpart of the third <<< >>> argument (ch05-facts C8).

bool encodeScanScanAdd(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> scanSegments,
                       id<MTLComputePipelineState> addBlockSums,
                       id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> scratch,
                       uint32_t n, uint32_t block) {
    if (!fits(scanSegments, block, true) || !fits(addBlockSums, block, false)) return false;
    // TODO 7: as launchScanScanAdd: dispatches of scan_segments and add_block_sums, one
    //         threadgroup of `block` threads per segment, recursing on the totals. A
    //         helper that takes (buffer, offset) pairs makes the recursion easy.
    return true;
}

bool encodeReduceScanScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> reduceSegments,
                          id<MTLComputePipelineState> scanWithCarry,
                          id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> scratch,
                          uint32_t n, uint32_t block) {
    if (!fits(reduceSegments, block, false) || !fits(scanWithCarry, block, true)) return false;
    // TODO 8: as launchReduceScanScan, with reduce_segments and scan_with_carry.
    return true;
}
