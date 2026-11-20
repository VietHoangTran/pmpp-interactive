// Lab 17.1 reference solution: the Metal counterparts of launchSpmvCsr and launchSpmvEll.
#import <Metal/Metal.h>
#include <cstdint>

#include "../spmv.h"   // SPMV_BLOCK

// SPMV_BLOCK threads per threadgroup, or fewer if this pipeline cannot run that many
// (its own limit may be below the device's: notes/today/ch05-facts.md C3).
static NSUInteger groupSize(id<MTLComputePipelineState> p) {
    return p.maxTotalThreadsPerThreadgroup < SPMV_BLOCK ? p.maxTotalThreadsPerThreadgroup : SPMV_BLOCK;
}

void encodeSpmvCsr(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   id<MTLBuffer> rowPtrs, id<MTLBuffer> colIdx, id<MTLBuffer> values,
                   id<MTLBuffer> x, id<MTLBuffer> y, uint32_t numRows) {
    if (numRows == 0) return;   // no rows, nothing to encode
    [enc setComputePipelineState:p];
    [enc setBuffer:rowPtrs offset:0 atIndex:0];
    [enc setBuffer:colIdx offset:0 atIndex:1];
    [enc setBuffer:values offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:y offset:0 atIndex:4];
    [enc setBytes:&numRows length:sizeof(numRows) atIndex:5];
    // One thread per row. dispatchThreads makes exactly numRows threads, the last
    // threadgroup partial if need be (ch05-facts C5).
    [enc dispatchThreads:MTLSizeMake(numRows, 1, 1) threadsPerThreadgroup:MTLSizeMake(groupSize(p), 1, 1)];
}

void encodeSpmvEll(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   id<MTLBuffer> colIdx, id<MTLBuffer> values, id<MTLBuffer> nnzPerRow,
                   id<MTLBuffer> x, id<MTLBuffer> y, uint32_t numRows, uint32_t width, uint32_t stride) {
    if (numRows == 0) return;
    [enc setComputePipelineState:p];
    [enc setBuffer:colIdx offset:0 atIndex:0];
    [enc setBuffer:values offset:0 atIndex:1];
    [enc setBuffer:nnzPerRow offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:y offset:0 atIndex:4];
    [enc setBytes:&numRows length:sizeof(numRows) atIndex:5];
    [enc setBytes:&width length:sizeof(width) atIndex:6];
    [enc setBytes:&stride length:sizeof(stride) atIndex:7];
    [enc dispatchThreads:MTLSizeMake(numRows, 1, 1) threadsPerThreadgroup:MTLSizeMake(groupSize(p), 1, 1)];
}
