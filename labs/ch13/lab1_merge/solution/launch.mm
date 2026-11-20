// Lab 13.1 reference solution: the Metal counterparts of the CUDA launchers.
// Each encode function returns false, without encoding anything, when this device
// cannot run it: a pipeline that allows fewer than MERGE_BLOCK threads per threadgroup,
// or (tiled) a tile that does not fit in the device's threadgroup memory.
#import <Metal/Metal.h>
#include <cstdint>

#include "../merge.h"

// Same layout as MergeParams in kernel.metal.
struct MergeParams { int32_t m, n, total, perThread; };

static void bindInputs(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> A, id<MTLBuffer> B, const MergeParams& p) {
    [enc setBuffer:A offset:0 atIndex:0];
    [enc setBuffer:B offset:0 atIndex:1];
    [enc setBytes:&p length:sizeof(p) atIndex:3];
}

// ---------------------------------------------------------------- Part (a)
bool encodeMergeBasic(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> A, uint32_t m, id<MTLBuffer> B, uint32_t n, id<MTLBuffer> C,
                      uint32_t total, uint32_t perThread) {
    if (pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)MERGE_BLOCK || perThread == 0) return false;
    if (total == 0) return true;                       // nothing to produce, nothing to dispatch
    MergeParams p = {(int32_t)m, (int32_t)n, (int32_t)total, (int32_t)perThread};
    [enc setComputePipelineState:pipeline];
    bindInputs(enc, A, B, p);
    [enc setBuffer:C offset:0 atIndex:2];
    const NSUInteger perGroup = (NSUInteger)MERGE_BLOCK * perThread;
    const NSUInteger groups = (total + perGroup - 1) / perGroup;    // integer ceiling division
    [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(MERGE_BLOCK, 1, 1)];
    return true;
}

// ---------------------------------------------------------------- Part (b)
// Two dispatches in one encoder. The encoder the runtime opens is the default serial
// kind, and buffers made by the device are hazard-tracked, so merge_tiled starts after
// merge_partition has written `bounds` (notes/today/ch02-facts.md C9).
bool encodeMergeTiled(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> partition,
                      id<MTLComputePipelineState> tiled, id<MTLBuffer> A, uint32_t m, id<MTLBuffer> B,
                      uint32_t n, id<MTLBuffer> C, uint32_t total, uint32_t perThread, id<MTLBuffer> bounds) {
    if (perThread == 0) return false;
    if (partition.maxTotalThreadsPerThreadgroup < (NSUInteger)MERGE_BLOCK ||
        tiled.maxTotalThreadsPerThreadgroup < (NSUInteger)MERGE_BLOCK) return false;
    const NSUInteger tileSize = (NSUInteger)MERGE_BLOCK * perThread;
    // 8 bytes per element: a multiple of 16 for any tile of 128 * perThread elements,
    // as setThreadgroupMemoryLength requires (ch05-facts C8).
    const NSUInteger tileBytes = tileSize * sizeof(KV);
    if (tileBytes + tiled.staticThreadgroupMemoryLength > enc.device.maxThreadgroupMemoryLength) return false;
    if (total == 0) return true;

    MergeParams p = {(int32_t)m, (int32_t)n, (int32_t)total, (int32_t)perThread};
    const NSUInteger numTiles = (total + tileSize - 1) / tileSize;

    [enc setComputePipelineState:partition];
    bindInputs(enc, A, B, p);
    [enc setBuffer:bounds offset:0 atIndex:2];
    const NSUInteger boundaries = numTiles + 1;
    [enc dispatchThreadgroups:MTLSizeMake((boundaries + MERGE_BLOCK - 1) / MERGE_BLOCK, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(MERGE_BLOCK, 1, 1)];

    [enc setComputePipelineState:tiled];
    bindInputs(enc, A, B, p);
    [enc setBuffer:C offset:0 atIndex:2];
    [enc setBuffer:bounds offset:0 atIndex:4];
    [enc setThreadgroupMemoryLength:tileBytes atIndex:0];   // matches [[threadgroup(0)]]
    [enc dispatchThreadgroups:MTLSizeMake(numTiles, 1, 1) threadsPerThreadgroup:MTLSizeMake(MERGE_BLOCK, 1, 1)];
    return true;
}

// ---------------------------------------------------------------- Provided
// out[k] = co_rank(k, A, m, B, n) for k = 0 .. m + n. Do not edit.
void encodeCoRank(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                  id<MTLBuffer> A, uint32_t m, id<MTLBuffer> B, uint32_t n, id<MTLBuffer> out) {
    MergeParams p = {(int32_t)m, (int32_t)n, (int32_t)(m + n), 1};
    [enc setComputePipelineState:pipeline];
    bindInputs(enc, A, B, p);
    [enc setBuffer:out offset:0 atIndex:2];
    const NSUInteger queries = m + n + 1;
    const NSUInteger width = pipeline.maxTotalThreadsPerThreadgroup < (NSUInteger)MERGE_BLOCK
                                 ? pipeline.maxTotalThreadsPerThreadgroup : (NSUInteger)MERGE_BLOCK;
    [enc dispatchThreadgroups:MTLSizeMake((queries + width - 1) / width, 1, 1)
        threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
}
