// Provided: encodes the lab 16.1 baseline on Metal, one dispatch per anti-diagonal.
//
// All dispatches go into the one compute encoder the driver opened. That encoder runs
// its dispatches in order, and buffers made from the device are hazard-tracked, so each
// anti-diagonal sees the previous ones' writes (notes/today/ch02-facts.md C9).
#import <Metal/Metal.h>
#include <algorithm>

struct SwBaselineParams { int L; int s; int firstRow; int count; };   // as in baseline.metal

void encodeSwBaseline(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> H, id<MTLBuffer> a, id<MTLBuffer> b, int Lseq) {
    const int group = (int)std::min<NSUInteger>(128, pipeline.maxTotalThreadsPerThreadgroup);
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:H offset:0 atIndex:0];
    [enc setBuffer:a offset:0 atIndex:1];
    [enc setBuffer:b offset:0 atIndex:2];
    for (int s = 2; s <= 2 * Lseq; s++) {                 // 2 * Lseq - 1 dispatches
        const int firstRow = std::max(1, s - Lseq);
        const int lastRow = std::min(Lseq, s - 1);
        SwBaselineParams p = {Lseq + 1, s, firstRow, lastRow - firstRow + 1};
        [enc setBytes:&p length:sizeof(p) atIndex:3];
        const NSUInteger groups = (NSUInteger)(p.count + group - 1) / group;
        [enc dispatchThreadgroups:MTLSizeMake(groups, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
    }
}
