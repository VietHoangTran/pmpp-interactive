// Lab 17.1 starter: the Metal counterparts of launchSpmvCsr and launchSpmvEll.
#import <Metal/Metal.h>
#include <cstdint>

#include "../spmv.h"   // SPMV_BLOCK

// A pipeline may accept fewer threads per threadgroup than SPMV_BLOCK
// (p.maxTotalThreadsPerThreadgroup); keep your threadgroup within both.

void encodeSpmvCsr(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   id<MTLBuffer> rowPtrs, id<MTLBuffer> colIdx, id<MTLBuffer> values,
                   id<MTLBuffer> x, id<MTLBuffer> y, uint32_t numRows) {
    // TODO 1: when numRows is 0, encode nothing at all.
    [enc setComputePipelineState:p];
    [enc setBuffer:rowPtrs offset:0 atIndex:0];
    [enc setBuffer:colIdx offset:0 atIndex:1];
    [enc setBuffer:values offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:y offset:0 atIndex:4];
    [enc setBytes:&numRows length:sizeof(numRows) atIndex:5];
    // TODO 2: one thread per row, threadgroups of at most SPMV_BLOCK threads:
    //   [enc dispatchThreads:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

void encodeSpmvEll(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   id<MTLBuffer> colIdx, id<MTLBuffer> values, id<MTLBuffer> nnzPerRow,
                   id<MTLBuffer> x, id<MTLBuffer> y, uint32_t numRows, uint32_t width, uint32_t stride) {
    // TODO 3: as TODO 1.
    [enc setComputePipelineState:p];
    [enc setBuffer:colIdx offset:0 atIndex:0];
    [enc setBuffer:values offset:0 atIndex:1];
    [enc setBuffer:nnzPerRow offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:y offset:0 atIndex:4];
    [enc setBytes:&numRows length:sizeof(numRows) atIndex:5];
    [enc setBytes:&width length:sizeof(width) atIndex:6];
    [enc setBytes:&stride length:sizeof(stride) atIndex:7];
    // TODO 4: as TODO 2.
}
