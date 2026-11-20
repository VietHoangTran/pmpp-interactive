// Lab 14.1 reference solution: the Metal host side.
//
//   runOddEvenSort      commits one command buffer per even-odd pair of steps, waits for it,
//                       then reads the flag on the CPU. A shared-storage buffer is visible to
//                       the CPU with no copy, but only after the GPU work that wrote it has
//                       completed (notes/today/ch05-facts.md C6; waitUntilCompleted blocks
//                       until then, infra-facts C12). That round trip per pair is the cost of
//                       this algorithm's host loop, on every backend.
//   encodeRadixSortOneBit puts all 32 passes, the provided scan's dispatches included, into the
//                       one compute encoder it is given. An encoder made with plain
//                       computeCommandEncoder runs its dispatches serially and each sees the
//                       previous one's writes (notes/today/ch02-facts.md C9): the Metal
//                       counterpart of 32 x 3 launches on one CUDA stream, with no host
//                       round trip at all.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as in kernel.cu

// Provided in ../../shared-g11/scan_launch.mm. It sets its own pipelines and buffers.
bool encodeExclusiveScan(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> data, uint32_t count, id<MTLBuffer> scratch);

static NSUInteger ceilDiv(NSUInteger a, NSUInteger b) { return (a + b - 1) / b; }

// ---------------------------------------------------------------- part A: odd-even sort

// Returns the number of steps launched, or -1 if a command buffer failed.
int runOddEvenSort(id<MTLCommandQueue> queue, id<MTLComputePipelineState> step, id<MTLBuffer> data,
                   id<MTLBuffer> flag, uint32_t n) {
    if (n < 2) {
        return 0;
    }
    const NSUInteger block = step.maxTotalThreadsPerThreadgroup < kBlock ? step.maxTotalThreadsPerThreadgroup : kBlock;
    const NSUInteger pairs = n / 2;
    volatile uint32_t* changed = (volatile uint32_t*)flag.contents;
    const int maxSteps = 2 * (int)n + 4;   // a correct sort never gets here; a broken one must not hang
    int steps = 0;
    while (steps < maxSteps) {
        @autoreleasepool {
            *changed = 0;   // no command buffer that uses the flag is in flight now
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:step];
            [enc setBuffer:data offset:0 atIndex:0];
            [enc setBuffer:flag offset:0 atIndex:1];
            [enc setBytes:&n length:sizeof(n) atIndex:2];
            for (uint32_t isOdd = 0; isOdd < 2; ++isOdd) {   // even step, then odd step
                [enc setBytes:&isOdd length:sizeof(isOdd) atIndex:3];
                [enc dispatchThreadgroups:MTLSizeMake(ceilDiv(pairs, block), 1, 1)
                    threadsPerThreadgroup:MTLSizeMake(block, 1, 1)];
            }
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            if (cb.status == MTLCommandBufferStatusError) {
                return -1;
            }
            steps += 2;
            if (*changed == 0) {
                break;   // a quiet pair: every adjacent pair is in order
            }
        }
    }
    return steps;
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
    if (n == 0) {
        return true;
    }
    const MTLSize threads = MTLSizeMake(kBlock, 1, 1);
    id<MTLBuffer> keysIn = keys, valsIn = vals, keysOut = tmpKeys, valsOut = tmpVals;
    for (uint32_t bit = 0; bit < 32; ++bit) {
        [enc setComputePipelineState:extract];
        [enc setBuffer:keysIn offset:0 atIndex:0];
        [enc setBuffer:bits offset:0 atIndex:1];
        [enc setBytes:&n length:sizeof(n) atIndex:2];
        [enc setBytes:&bit length:sizeof(bit) atIndex:3];
        [enc dispatchThreadgroups:MTLSizeMake(ceilDiv(n + 1, kBlock), 1, 1) threadsPerThreadgroup:threads];

        if (!encodeExclusiveScan(enc, bits, n + 1, scanScratch)) {   // n + 1 entries
            return false;
        }

        [enc setComputePipelineState:scatter];
        [enc setBuffer:keysIn offset:0 atIndex:0];
        [enc setBuffer:valsIn offset:0 atIndex:1];
        [enc setBuffer:keysOut offset:0 atIndex:2];
        [enc setBuffer:valsOut offset:0 atIndex:3];
        [enc setBuffer:bits offset:0 atIndex:4];
        [enc setBytes:&n length:sizeof(n) atIndex:5];
        [enc setBytes:&bit length:sizeof(bit) atIndex:6];
        [enc dispatchThreadgroups:MTLSizeMake(ceilDiv(n, kBlock), 1, 1) threadsPerThreadgroup:threads];

        id<MTLBuffer> swap = keysIn; keysIn = keysOut; keysOut = swap;
        swap = valsIn; valsIn = valsOut; valsOut = swap;
    }
    return true;   // 32 swaps: the result is back in keys/vals
}
