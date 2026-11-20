// Lab 11.6 starter: the Metal host side of the top-p keep mask. It binds the arguments and
// dispatches one threadgroup of kBlock threads per row. Return false, without dispatching,
// when the pipeline cannot run the kernel as written.
#import <Metal/Metal.h>
#include <cstdint>

static const NSUInteger kBlock = 256;   // threads per threadgroup, BLOCK_DIM in kernel.metal

bool encodeTopPMask(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                    id<MTLBuffer> probs, id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept,
                    uint32_t rows, uint32_t cols, float p) {
    // TODO 1: return false unless this pipeline can run kBlock threads per threadgroup in whole
    //         SIMD-groups, with no more SIMD-groups than a SIMD-group has lanes (your kernel's
    //         SIMD-group 0 gives one lane to each SIMD-group's total). The SIMD-group width is
    //         pipeline.threadExecutionWidth: it belongs to the compiled pipeline, so read it,
    //         never assume it. The pipeline's own thread limit is
    //         pipeline.maxTotalThreadsPerThreadgroup.
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:probs offset:0 atIndex:0];
    [enc setBuffer:massBefore offset:0 atIndex:1];
    [enc setBuffer:keep offset:0 atIndex:2];
    [enc setBuffer:kept offset:0 atIndex:3];
    [enc setBytes:&cols length:sizeof(cols) atIndex:4];
    [enc setBytes:&p length:sizeof(p) atIndex:5];
    // TODO 2: dispatch one threadgroup of kBlock threads per row.
    return true;
}
