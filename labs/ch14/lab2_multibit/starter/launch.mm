// Lab 14.2 starter: the Metal counterpart of launchRadixSortMulti.
// Put every pass (local sort, the provided scan of the table, scatter) into the one compute
// encoder you are given. An encoder from plain computeCommandEncoder runs its dispatches in
// order and each sees the previous one's writes (notes/today/ch02-facts.md C9).
//
// Threadgroup memory for radix_local_sort: the tile (256c keys and 256c values) at
// [[threadgroup(0)]] and one slot per SIMD-group at [[threadgroup(1)]], set with
// setThreadgroupMemoryLength:atIndex: in multiples of 16 bytes, plus whatever the kernel
// declares itself (the pipeline's staticThreadgroupMemoryLength). The sum must fit the
// device's maxThreadgroupMemoryLength (notes/today/ch05-facts.md C1, C3, C8).
//
// Return false, before encoding anything, if this configuration cannot run on this device.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;          // BLOCK_DIM in kernel.metal
static const uint32_t kMaxKeysPerThread = 8;   // MAX_KEYS_PER_THREAD
static const uint32_t kMaxRadixBits = 8;       // MAX_RADIX_BITS

// Provided in ../../shared-g11/scan_launch.mm. In place, `count` entries. It sets its own
// pipelines and buffers, so set everything again for the dispatch after it.
bool encodeExclusiveScan(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> data, uint32_t count, id<MTLBuffer> scratch);

static NSUInteger roundUp16(NSUInteger bytes) { return (bytes + 15) / 16 * 16; }

bool encodeRadixSortMulti(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> localSort,
                          id<MTLComputePipelineState> scatterLocal, id<MTLBuffer> keys, id<MTLBuffer> vals,
                          id<MTLBuffer> tmpKeys, id<MTLBuffer> tmpVals, id<MTLBuffer> table,
                          id<MTLBuffer> localStarts, id<MTLBuffer> scanScratch, uint32_t n, uint32_t radixBits,
                          uint32_t keysPerThread) {
    if (radixBits < 1 || radixBits > kMaxRadixBits || keysPerThread < 1 || keysPerThread > kMaxKeysPerThread) {
        return false;
    }
    // TODO 1: read the SIMD-group width from the pipeline (threadExecutionWidth); refuse a
    //         width that does not divide kBlock, and pipelines that cannot run kBlock threads.
    // TODO 2: compute the two threadgroup-memory lengths and refuse what does not fit
    //         enc.device.maxThreadgroupMemoryLength.
    // TODO 3: the passes, as launchRadixSortMulti in kernel.cu: bind radix_local_sort's buffers
    //         and constants (indices as in kernel.metal), set the threadgroup-memory lengths,
    //         dispatch; the provided scan; bind and dispatch radix_scatter_local.
    return true;
}
