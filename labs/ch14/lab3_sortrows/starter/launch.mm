// Lab 14.3 starter: the Metal counterpart of launchSortRowsDesc. It encodes the whole sort
// into the one compute encoder it is given: desc_keys, then one sort_rows_pass per digit,
// then gather_rows, each one threadgroup of kBlock threads per row. Return false, without
// dispatching, when a pipeline cannot run the kernels as written.
//
// A plain compute encoder dispatches serially, and buffers made by the device are
// hazard-tracked, so each dispatch sees the writes of the one before
// (notes/today/ch02-facts.md C9). You need no extra synchronisation between the passes.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;     // BLOCK_SIZE in kernel.metal
static const uint32_t kRadixBits = 8;     // RADIX_BITS in kernel.metal: keep the two equal

// TODO 1: a check that sortPass can run kBlock threads per threadgroup in whole SIMD-groups,
//         with no more SIMD-groups than a SIMD-group has lanes (the provided scan needs both),
//         and that its threadgroup memory fits: the arrays the kernel declares
//         (p.staticThreadgroupMemoryLength) plus group_sums, within
//         p.device.maxThreadgroupMemoryLength. The SIMD-group width is p.threadExecutionWidth:
//         it belongs to the compiled pipeline, so read it, never assume it.

bool encodeSortRowsDesc(id<MTLComputeCommandEncoder> enc,
                        id<MTLComputePipelineState> descKeys,
                        id<MTLComputePipelineState> sortPass,
                        id<MTLComputePipelineState> gather,
                        id<MTLBuffer> probs, id<MTLBuffer> sortedProbs, id<MTLBuffer> sortedIdx,
                        id<MTLBuffer> scratch, uint32_t rows, uint32_t cols) {
    // TODO 2: scratch holds four regions of rows * cols uints: two of keys, two of indices.
    //         Bind a region with [enc setBuffer:scratch offset:bytes atIndex:i]; a device
    //         buffer's offset must be a multiple of the element's size (ch06-facts C3).
    //         Then, with the binding indices of kernel.metal:
    //           desc_keys once;
    //           sort_rows_pass once per digit (shift = 0, kRadixBits, ...), each reading the
    //           regions the previous dispatch wrote. Set the shift with setBytes; size
    //           group_sums (one uint per SIMD-group, rounded up to a multiple of 16 bytes,
    //           ch05-facts C8) with [enc setThreadgroupMemoryLength:bytes atIndex:0];
    //           gather_rows from the index region the last pass wrote.
    //         Every dispatch: dispatchThreadgroups of (rows, 1, 1) with (kBlock, 1, 1) threads.
    return true;
}
