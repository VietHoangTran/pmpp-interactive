// Lab 13.1 starter: the Metal counterparts of the CUDA launchers.
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
    // TODO 1: return false if this pipeline cannot run MERGE_BLOCK threads per
    //         threadgroup (pipeline.maxTotalThreadsPerThreadgroup).
    // TODO 2: nothing to dispatch when total is 0 (that is not a refusal).
    // TODO 3: set the pipeline, bind A, B and the parameters (bindInputs), C at index 2,
    //         and dispatch threadgroups of MERGE_BLOCK threads, as launchMergeBasic does.
    return true;
}

// ---------------------------------------------------------------- Part (b)
// Two dispatches in one encoder: merge_partition, then merge_tiled. The encoder the
// runtime opens is the default serial kind, and buffers made by the device are
// hazard-tracked, so the second dispatch starts after the first has written `bounds`
// (notes/today/ch02-facts.md C9).
bool encodeMergeTiled(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> partition,
                      id<MTLComputePipelineState> tiled, id<MTLBuffer> A, uint32_t m, id<MTLBuffer> B,
                      uint32_t n, id<MTLBuffer> C, uint32_t total, uint32_t perThread, id<MTLBuffer> bounds) {
    // TODO 4: return false if either pipeline cannot run MERGE_BLOCK threads per
    //         threadgroup, or if the tile's bytes (plus tiled.staticThreadgroupMemoryLength)
    //         exceed enc.device.maxThreadgroupMemoryLength.
    // TODO 5: nothing to dispatch when total is 0.
    // TODO 6: dispatch merge_partition (bounds at index 2), then merge_tiled (C at index 2,
    //         bounds at index 4). Before the second dispatch, size [[threadgroup(0)]] with
    //         [enc setThreadgroupMemoryLength:bytes atIndex:0]; the length must be a
    //         multiple of 16 bytes (notes/today/ch05-facts.md C8).
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
