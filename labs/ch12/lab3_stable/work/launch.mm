// Lab 12.3 starter: the Metal counterparts of launchStableFilterFlags and
// launchStableFilterStaged. Each encodes several dispatches, the provided scan in the middle,
// into the one compute encoder it is given. An encoder from plain computeCommandEncoder runs
// its dispatches in order and each sees the previous one's writes (notes/today/ch02-facts.md
// C9), the Metal counterpart of three kernel launches on one CUDA stream.
//
// Each returns false if a pipeline cannot run the kernels as written; the driver then reports
// why. Return true once you have encoded everything.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, BLOCK_DIM in kernel.cu

// Provided in ../../shared-g12/scan_u32_launch.mm. It sets its own pipeline and buffers, so set
// everything again for the dispatch after it.
bool encodeExclusiveScanU32(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n,
                            id<MTLBuffer> scratch);

static NSUInteger ceilDiv(NSUInteger a, NSUInteger b) { return (a + b - 1) / b; }

bool encodeStableFilterFlags(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> keepFlags,
                             id<MTLComputePipelineState> scatterFlags, id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize, id<MTLBuffer> keep,
                             id<MTLBuffer> offset, id<MTLBuffer> scanScratch) {
    if (kBlock > keepFlags.maxTotalThreadsPerThreadgroup || kBlock > scatterFlags.maxTotalThreadsPerThreadgroup) {
        return false;
    }
    // 1. keep flags: bind what keep_flags takes, then dispatch enough threadgroups of kBlock
    //    threads to cover n keys. For example:
    //      [enc setComputePipelineState:keepFlags];
    //      [enc setBuffer:input offset:0 atIndex:0];
    //      [enc setBytes:&n length:sizeof(n) atIndex:2];
    //      ...
    //      [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
    // TODO A5: encode step 1.

    // TODO A6: encode step 2, the provided scan of keep into offset (check what it returns).

    // TODO A7: encode step 3, scatter_flags.
    return true;
}

bool encodeStableFilterStaged(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> countKept,
                              id<MTLComputePipelineState> stagedScatter, id<MTLBuffer> input, id<MTLBuffer> output,
                              uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize, id<MTLBuffer> blockCounts,
                              id<MTLBuffer> blockOffsets, id<MTLBuffer> scanScratch, uint32_t coarse) {
    // TODO B8: check that both pipelines run kBlock threads per threadgroup in whole SIMD-groups.
    //          The SIMD-group width is the pipeline's threadExecutionWidth.

    // TODO B9: size the threadgroup memory. Lengths must be multiples of 16 bytes (ch05-facts C8).
    //          groupTotals: one uint per SIMD-group. staged_s: kBlock * coarse uints.
    //          If staged_scatter's static threadgroup memory plus both does not fit in the
    //          device's maxThreadgroupMemoryLength (enc.device), return false.

    // TODO B10: one threadgroup per tile of kBlock * coarse keys (round up). Encode count_kept
    //           (setThreadgroupMemoryLength:atIndex: for its groupTotals), the provided scan of
    //           blockCounts into blockOffsets, then staged_scatter with both of its
    //           threadgroup buffers.
    return true;
}
