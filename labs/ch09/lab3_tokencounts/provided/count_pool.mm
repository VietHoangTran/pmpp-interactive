// Lab 9.3, provided (do not edit): the Metal host side of provided/count_pool.metal.
#import <Metal/Metal.h>
#include <cstdint>

static const uint32_t kPoolGroupSize = 256;

uint32_t countPoolBlocksPerRow(uint32_t len) { return (len + kPoolGroupSize - 1) / kPoolGroupSize; }

// Zeroes the pool, then runs Fig. 9.9's kernel. The compute encoder the lab runtime opens
// is serial, so the second dispatch starts after the first has finished and sees its
// writes (notes/today/ch02-facts.md C9). Returns false,
// dispatching nothing, if either pipeline cannot run 256 threads per threadgroup.
bool encodeCountPool(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> zero,
                     id<MTLComputePipelineState> count, id<MTLBuffer> tokens, id<MTLBuffer> counts,
                     id<MTLBuffer> pool, uint32_t batch, uint32_t len, uint32_t vocab) {
    if (zero.maxTotalThreadsPerThreadgroup < kPoolGroupSize ||
        count.maxTotalThreadsPerThreadgroup < kPoolGroupSize) return false;
    const uint32_t blocksPerRow = countPoolBlocksPerRow(len);
    const uint32_t poolSize = batch * blocksPerRow * vocab;

    [enc setComputePipelineState:zero];
    [enc setBuffer:pool offset:0 atIndex:0];
    [enc setBytes:&poolSize length:sizeof(poolSize) atIndex:1];
    [enc dispatchThreadgroups:MTLSizeMake((poolSize + kPoolGroupSize - 1) / kPoolGroupSize, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(kPoolGroupSize, 1, 1)];

    [enc setComputePipelineState:count];
    [enc setBuffer:tokens offset:0 atIndex:0];
    [enc setBuffer:counts offset:0 atIndex:1];
    [enc setBuffer:pool offset:0 atIndex:2];
    [enc setBytes:&len length:sizeof(len) atIndex:3];
    [enc setBytes:&vocab length:sizeof(vocab) atIndex:4];
    [enc dispatchThreadgroups:MTLSizeMake(blocksPerRow, batch, 1)
        threadsPerThreadgroup:MTLSizeMake(kPoolGroupSize, 1, 1)];
    return true;
}
