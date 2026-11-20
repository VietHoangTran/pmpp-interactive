// Lab 14.2 reference solution: the Metal counterpart of launchRadixSortMulti.
// Every pass (local sort, the provided scan of the table, scatter) goes into the one compute
// encoder it is given. An encoder from plain computeCommandEncoder runs its dispatches in order
// and each sees the previous one's writes (notes/today/ch02-facts.md C9).
//
// Threadgroup memory for radix_local_sort: the tile, 256c keys and 256c values (8 bytes per
// pair), at [[threadgroup(0)]]; one slot per SIMD-group at [[threadgroup(1)]]; plus the
// kernel's static bucket-start array, which the pipeline reports as
// staticThreadgroupMemoryLength. Lengths set here must be multiples of 16 bytes, and the sum
// must fit the device's maxThreadgroupMemoryLength (notes/today/ch05-facts.md C1, C3, C8).
//
// Returns false if a configuration cannot run on this device; that is checked before anything
// is encoded.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;          // BLOCK_DIM in kernel.metal
static const uint32_t kMaxKeysPerThread = 8;   // MAX_KEYS_PER_THREAD
static const uint32_t kMaxRadixBits = 8;       // MAX_RADIX_BITS

// Provided in ../../shared-g11/scan_launch.mm. It sets its own pipelines and buffers.
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
    // Whole SIMD-groups of a width read from the pipeline, never assumed.
    const NSUInteger width = localSort.threadExecutionWidth;
    if (width == 0 || kBlock % width != 0) {
        return false;
    }
    if (kBlock > localSort.maxTotalThreadsPerThreadgroup || kBlock > scatterLocal.maxTotalThreadsPerThreadgroup) {
        return false;
    }
    const NSUInteger tileBytes = roundUp16(2 * kBlock * keysPerThread * sizeof(uint32_t));
    const NSUInteger totalsBytes = roundUp16(kBlock / width * sizeof(uint32_t));
    if (localSort.staticThreadgroupMemoryLength + tileBytes + totalsBytes > enc.device.maxThreadgroupMemoryLength) {
        return false;
    }
    if (n == 0) {
        return true;
    }
    const uint32_t tile = (uint32_t)kBlock * keysPerThread;
    const uint32_t numBlocks = (n + tile - 1) / tile;
    const MTLSize groups = MTLSizeMake(numBlocks, 1, 1);
    const MTLSize threads = MTLSizeMake(kBlock, 1, 1);

    for (uint32_t shift = 0; shift < 32; shift += radixBits) {
        const uint32_t bits = 32 - shift < radixBits ? 32 - shift : radixBits;   // the last pass may be narrower

        // 1. local sort: keys/vals -> tmpKeys/tmpVals, the table and the local starts
        [enc setComputePipelineState:localSort];
        [enc setBuffer:keys offset:0 atIndex:0];
        [enc setBuffer:vals offset:0 atIndex:1];
        [enc setBuffer:tmpKeys offset:0 atIndex:2];
        [enc setBuffer:tmpVals offset:0 atIndex:3];
        [enc setBuffer:table offset:0 atIndex:4];
        [enc setBuffer:localStarts offset:0 atIndex:5];
        [enc setBytes:&n length:sizeof(n) atIndex:6];
        [enc setBytes:&shift length:sizeof(shift) atIndex:7];
        [enc setBytes:&bits length:sizeof(bits) atIndex:8];
        [enc setBytes:&keysPerThread length:sizeof(keysPerThread) atIndex:9];
        [enc setThreadgroupMemoryLength:tileBytes atIndex:0];
        [enc setThreadgroupMemoryLength:totalsBytes atIndex:1];
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];

        // 2. global bucket starts: the provided scan of the 2^bits x numBlocks table
        if (!encodeExclusiveScan(enc, table, (1u << bits) * numBlocks, scanScratch)) {
            return false;
        }

        // 3. scatter: tmpKeys/tmpVals -> keys/vals
        [enc setComputePipelineState:scatterLocal];
        [enc setBuffer:tmpKeys offset:0 atIndex:0];
        [enc setBuffer:tmpVals offset:0 atIndex:1];
        [enc setBuffer:keys offset:0 atIndex:2];
        [enc setBuffer:vals offset:0 atIndex:3];
        [enc setBuffer:table offset:0 atIndex:4];
        [enc setBuffer:localStarts offset:0 atIndex:5];
        [enc setBytes:&n length:sizeof(n) atIndex:6];
        [enc setBytes:&shift length:sizeof(shift) atIndex:7];
        [enc setBytes:&bits length:sizeof(bits) atIndex:8];
        [enc setBytes:&keysPerThread length:sizeof(keysPerThread) atIndex:9];
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    }
    return true;
}
