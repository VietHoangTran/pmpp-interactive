// Lab 11.4 reference solution: the Metal counterparts of the CUDA launchers.
//
// A whole global scan is several dispatches written into one compute encoder that the
// caller opened and will close, commit and wait for. Each dispatch must see what the one
// before it wrote, and Apple documents two reasons it does (ch02-facts C9): the plain
// computeCommandEncoder "dispatches its compute commands serially", and buffers created
// from the device are hazard-tracked by default, so Metal will "prevent subsequent
// commands from running until write operations finish". No threadgroup ever waits for
// another one inside a dispatch: Metal documents no forward progress between
// threadgroups (ch11-facts C5), and the hierarchical scan does not need any.
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

// One float per segment at every level, as in kernel.cu.
uint32_t scratchFloats(uint32_t n, uint32_t block) {
    uint32_t total = 0;
    do {
        n = segmentsOf(n, block);
        total += n;
    } while (n > 1);
    return total;
}

// The block must fit the pipeline (ch05-facts C3) and be whole SIMD-groups, at most
// `width` of them (stage 2 of the block scan and of the reduction runs in one
// SIMD-group). With a segment buffer, the threadgroup memory must fit the device
// (ch05-facts C3, C8).
static bool fits(id<MTLComputePipelineState> p, uint32_t block, bool segmentBuffer) {
    if (p == nil || block == 0 || block > p.maxTotalThreadsPerThreadgroup) return false;
    const NSUInteger width = p.threadExecutionWidth;
    if (block % width != 0 || block / width > width) return false;
    const NSUInteger extra = segmentBuffer ? (NSUInteger)COARSE_FACTOR * block * sizeof(float) : 0;
    return p.staticThreadgroupMemoryLength + extra <= p.device.maxThreadgroupMemoryLength;
}

// A float array that lives at a byte offset inside a buffer. setBuffer:offset: wants
// the offset aligned to the element's size (ch06-facts C3): floats, so 4 bytes.
struct Floats {
    id<MTLBuffer> buf;
    NSUInteger offset;
    Floats after(uint32_t count) const { return Floats{buf, offset + (NSUInteger)count * sizeof(float)}; }
};

static void dispatchSegments(id<MTLComputeCommandEncoder> enc, uint32_t blocks, uint32_t block,
                             bool segmentBuffer) {
    if (segmentBuffer) {
        // The counterpart of the third <<< >>> argument. It must be a multiple of 16
        // bytes (ch05-facts C8); at COARSE_FACTOR 4 it is 16 bytes per thread.
        [enc setThreadgroupMemoryLength:COARSE_FACTOR * block * sizeof(float) atIndex:0];
    }
    [enc dispatchThreadgroups:MTLSizeMake(blocks, 1, 1) threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
}

// ---------------------------------------------------------------- Fig. 11.14
static void scanScanAdd(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> scanSegments,
                        id<MTLComputePipelineState> addBlockSums,
                        Floats in, Floats out, Floats scratch, uint32_t n, uint32_t block) {
    const uint32_t blocks = segmentsOf(n, block);
    const Floats blockSums = scratch;
    // Kernel 1: scan each segment, totals into blockSums.
    [enc setComputePipelineState:scanSegments];
    [enc setBuffer:in.buf offset:in.offset atIndex:0];
    [enc setBuffer:out.buf offset:out.offset atIndex:1];
    [enc setBuffer:blockSums.buf offset:blockSums.offset atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    dispatchSegments(enc, blocks, block, true);
    if (blocks == 1) return;
    // Kernel 2: the same design on the totals, in place.
    scanScanAdd(enc, scanSegments, addBlockSums, blockSums, blockSums, scratch.after(blocks), blocks, block);
    // Kernel 3: add the earlier segments' total.
    [enc setComputePipelineState:addBlockSums];
    [enc setBuffer:out.buf offset:out.offset atIndex:0];
    [enc setBuffer:blockSums.buf offset:blockSums.offset atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    dispatchSegments(enc, blocks, block, false);
}

bool encodeScanScanAdd(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> scanSegments,
                       id<MTLComputePipelineState> addBlockSums,
                       id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> scratch,
                       uint32_t n, uint32_t block) {
    if (!fits(scanSegments, block, true) || !fits(addBlockSums, block, false)) return false;
    scanScanAdd(enc, scanSegments, addBlockSums, Floats{in, 0}, Floats{out, 0}, Floats{scratch, 0}, n, block);
    return true;
}

// ---------------------------------------------------------------- Fig. 11.15
static void reduceScanScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> reduceSegments,
                           id<MTLComputePipelineState> scanWithCarry,
                           Floats in, Floats out, Floats scratch, uint32_t n, uint32_t block) {
    const uint32_t blocks = segmentsOf(n, block);
    const Floats blockSums = scratch;
    if (blocks > 1) {
        // Kernel 1: each segment's total.
        [enc setComputePipelineState:reduceSegments];
        [enc setBuffer:in.buf offset:in.offset atIndex:0];
        [enc setBuffer:blockSums.buf offset:blockSums.offset atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        dispatchSegments(enc, blocks, block, false);
        // Kernel 2: scan the totals, in place, with the same design.
        reduceScanScan(enc, reduceSegments, scanWithCarry, blockSums, blockSums, scratch.after(blocks), blocks, block);
    }
    // Kernel 3: carry into the first element, scan. With one segment, threadgroup 0 never
    // reads blockSums.
    [enc setComputePipelineState:scanWithCarry];
    [enc setBuffer:in.buf offset:in.offset atIndex:0];
    [enc setBuffer:out.buf offset:out.offset atIndex:1];
    [enc setBuffer:blockSums.buf offset:blockSums.offset atIndex:2];
    [enc setBytes:&n length:sizeof(n) atIndex:3];
    dispatchSegments(enc, blocks, block, true);
}

bool encodeReduceScanScan(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> reduceSegments,
                          id<MTLComputePipelineState> scanWithCarry,
                          id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> scratch,
                          uint32_t n, uint32_t block) {
    if (!fits(reduceSegments, block, false) || !fits(scanWithCarry, block, true)) return false;
    reduceScanScan(enc, reduceSegments, scanWithCarry, Floats{in, 0}, Floats{out, 0}, Floats{scratch, 0}, n, block);
    return true;
}
