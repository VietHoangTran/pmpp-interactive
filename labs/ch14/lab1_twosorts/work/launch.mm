// Lab 14.1 starter: the Metal host side.
//
//   runOddEvenSort        the odd-even host loop. Commit one command buffer per even-odd pair
//                         of steps, wait for it, then read the flag on the CPU. A shared-storage
//                         buffer is visible to the CPU with no copy, but only after the GPU work
//                         that wrote it has completed (notes/today/ch05-facts.md C6;
//                         waitUntilCompleted blocks until then, infra-facts C12).
//   encodeRadixSortOneBit puts all 32 radix passes, the provided scan's dispatches included,
//                         into the one compute encoder it is given. An encoder made with plain
//                         computeCommandEncoder runs its dispatches serially and each sees the
//                         previous one's writes (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as in kernel.cu

// Provided in ../../shared-g11/scan_launch.mm. In place, n entries: data[0] = 0, data[i] = the
// sum of the old data[0..i-1]. It sets its own pipelines and buffers, so set everything again
// for the dispatch after it. Returns false if it cannot run.
bool encodeExclusiveScan(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> data, uint32_t count, id<MTLBuffer> scratch);

static NSUInteger ceilDiv(NSUInteger a, NSUInteger b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------- part A: odd-even sort

// Returns the number of steps launched, or -1 if a command buffer failed.
int runOddEvenSort(id<MTLCommandQueue> queue, id<MTLComputePipelineState> step, id<MTLBuffer> data,
                   id<MTLBuffer> flag, uint32_t n) {
    // TODO: the loop of launchOddEvenSort in kernel.cu, in Metal's vocabulary. For each pair:
    //   - clear the flag through flag.contents (safe: nothing that uses it is in flight);
    //   - [queue commandBuffer], then [cb computeCommandEncoder]; bind odd_even_step's buffers
    //     and constants (setBuffer:offset:atIndex:, setBytes:length:atIndex:) and dispatch the
    //     even step and the odd step (dispatchThreadgroups:threadsPerThreadgroup:, with at most
    //     step.maxTotalThreadsPerThreadgroup threads per threadgroup);
    //   - endEncoding, commit, waitUntilCompleted; check cb.status;
    //   - read the flag and decide.
    // Same guards as in kernel.cu: nothing to do below 2 elements; give up after 2n + 4 steps.
    // Wrap each iteration in @autoreleasepool { }, as the course runtime does around each of its
    // command buffers (labs/common/src/backend_metal.mm, pmpp::run).
    return 0;
}

// ---------------------------------------------------------------- part B: one-bit radix sort

// Returns false if a pipeline cannot run kBlock threads per threadgroup or the scan cannot run.
bool encodeRadixSortOneBit(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> extract,
                           id<MTLComputePipelineState> scatter, id<MTLBuffer> keys, id<MTLBuffer> vals,
                           id<MTLBuffer> tmpKeys, id<MTLBuffer> tmpVals, id<MTLBuffer> bits,
                           id<MTLBuffer> scanScratch, uint32_t n) {
    if (kBlock > extract.maxTotalThreadsPerThreadgroup || kBlock > scatter.maxTotalThreadsPerThreadgroup) {
        return false;
    }
    // TODO: for each of the 32 bits, encode radix_extract_bits, the provided scan and
    //       radix_scatter, as launchRadixSortOneBit does in kernel.cu. For example:
    //         [enc setComputePipelineState:extract];
    //         [enc setBuffer:... offset:0 atIndex:0];
    //         [enc setBytes:&n length:sizeof(n) atIndex:2];
    //         [enc dispatchThreadgroups:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(kBlock, 1, 1)];
    return true;
}
