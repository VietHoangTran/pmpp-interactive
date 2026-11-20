// Lab 12.4 starter: the Metal counterpart of launchCompactCandidates. Binds the arguments,
// sizes the threadgroup memory the kernel declares with [[threadgroup(0)]], and dispatches
// one threadgroup of kBlock threads per row. Return false, without dispatching, when the
// pipeline cannot run the kernel as the lab specifies it.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, as BLOCK_SIZE in kernel.cu

// TODO 1: a check that this pipeline can run kBlock threads per threadgroup in whole
//         SIMD-groups, with no more SIMD-groups than a SIMD-group has lanes. The SIMD-group
//         width is p.threadExecutionWidth: it belongs to the compiled pipeline, so read it,
//         never assume it. The pipeline's own thread limit is p.maxTotalThreadsPerThreadgroup.

bool encodeCompactCandidates(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                             id<MTLBuffer> logits, id<MTLBuffer> threshold, id<MTLBuffer> outIdx,
                             id<MTLBuffer> outVal, id<MTLBuffer> counts, uint32_t rows, uint32_t cols) {
    [enc setComputePipelineState:p];
    [enc setBuffer:logits offset:0 atIndex:0];
    [enc setBuffer:threshold offset:0 atIndex:1];
    [enc setBuffer:outIdx offset:0 atIndex:2];
    [enc setBuffer:outVal offset:0 atIndex:3];
    [enc setBuffer:counts offset:0 atIndex:4];
    [enc setBytes:&cols length:sizeof(cols) atIndex:5];
    // TODO 2: size groupTotals (one uint per SIMD-group) with
    //         [enc setThreadgroupMemoryLength:bytes atIndex:0]; the length must be a
    //         multiple of 16 bytes. Then dispatch one threadgroup per row.
    return true;
}
