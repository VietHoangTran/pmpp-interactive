// Lab 12.3 reference solution: the Metal counterparts of launchStableFilterFlags and
// launchStableFilterStaged. Each encodes several dispatches, the provided scan in the middle,
// into the one compute encoder it is given. An encoder from plain computeCommandEncoder runs
// its dispatches in order and each sees the previous one's writes (notes/today/ch02-facts.md
// C9), the Metal counterpart of three kernel launches on one CUDA stream.
//
// Each returns false if a pipeline cannot run the kernels as written; the driver then reports
// why.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, BLOCK_DIM in kernel.cu

// Provided in ../../shared-g12/scan_u32_launch.mm.
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
    const MTLSize groups = MTLSizeMake(ceilDiv(n, kBlock), 1, 1);
    const MTLSize threads = MTLSizeMake(kBlock, 1, 1);

    // 1. keep flags
    [enc setComputePipelineState:keepFlags];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:keep offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&threshold length:sizeof(threshold) atIndex:3];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];

    // 2. offsets: the provided scan (it binds its own pipelines and buffers)
    if (!encodeExclusiveScanU32(enc, keep, offset, n, scanScratch)) {
        return false;
    }

    // 3. scatter
    [enc setComputePipelineState:scatterFlags];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:keep offset:0 atIndex:1];
    [enc setBuffer:offset offset:0 atIndex:2];
    [enc setBuffer:output offset:0 atIndex:3];
    [enc setBuffer:outputSize offset:0 atIndex:4];
    [enc setBytes:&n length:sizeof(n) atIndex:5];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    return true;
}

bool encodeStableFilterStaged(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> countKept,
                              id<MTLComputePipelineState> stagedScatter, id<MTLBuffer> input, id<MTLBuffer> output,
                              uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize, id<MTLBuffer> blockCounts,
                              id<MTLBuffer> blockOffsets, id<MTLBuffer> scanScratch, uint32_t coarse) {
    // The SIMD-group width is a property of the compiled pipeline (threadExecutionWidth). The
    // kernels need whole SIMD-groups and one threadgroup-memory slot per SIMD-group.
    const NSUInteger width = countKept.threadExecutionWidth;
    if (width == 0 || width != stagedScatter.threadExecutionWidth || kBlock % width != 0 || coarse == 0) {
        return false;
    }
    if (kBlock > countKept.maxTotalThreadsPerThreadgroup || kBlock > stagedScatter.maxTotalThreadsPerThreadgroup) {
        return false;
    }
    // Threadgroup memory lengths must be multiples of 16 bytes (ch05-facts C8).
    const NSUInteger totalsBytes = ceilDiv(kBlock / width * sizeof(uint32_t), 16) * 16;
    const NSUInteger stagingBytes = ceilDiv(kBlock * coarse * sizeof(uint32_t), 16) * 16;
    const NSUInteger maxBytes = enc.device.maxThreadgroupMemoryLength;
    if (stagedScatter.staticThreadgroupMemoryLength + stagingBytes + totalsBytes > maxBytes) {
        return false;   // this coarsening factor does not fit this device
    }
    const uint32_t tiles = (uint32_t)ceilDiv(n, kBlock * coarse);
    const MTLSize groups = MTLSizeMake(tiles, 1, 1);
    const MTLSize threads = MTLSizeMake(kBlock, 1, 1);

    // 1. count the kept keys of every tile
    [enc setComputePipelineState:countKept];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:blockCounts offset:0 atIndex:1];
    [enc setBytes:&n length:sizeof(n) atIndex:2];
    [enc setBytes:&threshold length:sizeof(threshold) atIndex:3];
    [enc setBytes:&coarse length:sizeof(coarse) atIndex:4];
    [enc setThreadgroupMemoryLength:totalsBytes atIndex:0];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];

    // 2. block offsets: the provided scan of the counts
    if (!encodeExclusiveScanU32(enc, blockCounts, blockOffsets, tiles, scanScratch)) {
        return false;
    }

    // 3. stage and write
    [enc setComputePipelineState:stagedScatter];
    [enc setBuffer:input offset:0 atIndex:0];
    [enc setBuffer:output offset:0 atIndex:1];
    [enc setBuffer:blockOffsets offset:0 atIndex:2];
    [enc setBuffer:outputSize offset:0 atIndex:3];
    [enc setBytes:&n length:sizeof(n) atIndex:4];
    [enc setBytes:&threshold length:sizeof(threshold) atIndex:5];
    [enc setBytes:&coarse length:sizeof(coarse) atIndex:6];
    [enc setThreadgroupMemoryLength:stagingBytes atIndex:0];
    [enc setThreadgroupMemoryLength:totalsBytes atIndex:1];
    [enc dispatchThreadgroups:groups threadsPerThreadgroup:threads];
    return true;
}
