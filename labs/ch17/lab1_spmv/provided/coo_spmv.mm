// Lab 17.1, provided (do not edit): the Metal host side of provided/coo_spmv.metal.
#import <Metal/Metal.h>
#include <cstdint>

#include "../spmv.h"

// One thread per entry, in threadgroups of SPMV_BLOCK threads or fewer if this pipeline
// cannot run that many (its limit may be below the device's: notes/today/ch05-facts.md C3).
// Encodes nothing when there are no entries. In the hybrid, the lab encodes this before the
// ELL kernel in the same compute pass; that pass is serial, so the ELL kernel starts after
// this one has finished and sees its writes (notes/today/ch02-facts.md C9).
void encodeSpmvCoo(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p,
                   id<MTLBuffer> rowIdx, id<MTLBuffer> colIdx, id<MTLBuffer> values,
                   id<MTLBuffer> x, id<MTLBuffer> y, uint32_t numNonzeros) {
    if (numNonzeros == 0) return;
    [enc setComputePipelineState:p];
    [enc setBuffer:rowIdx offset:0 atIndex:0];
    [enc setBuffer:colIdx offset:0 atIndex:1];
    [enc setBuffer:values offset:0 atIndex:2];
    [enc setBuffer:x offset:0 atIndex:3];
    [enc setBuffer:y offset:0 atIndex:4];
    [enc setBytes:&numNonzeros length:sizeof(numNonzeros) atIndex:5];
    const NSUInteger group = p.maxTotalThreadsPerThreadgroup < SPMV_BLOCK ? p.maxTotalThreadsPerThreadgroup : SPMV_BLOCK;
    [enc dispatchThreads:MTLSizeMake(numNonzeros, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
}
