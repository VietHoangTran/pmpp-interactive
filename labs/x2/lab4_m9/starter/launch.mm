// Lab X2.4 starter: the Metal host side. Not in the book.
//
// encodeEmbedQ, encodeQLinear and encodeQDecodeStep are the counterparts of launchEmbedQ,
// launchQLinear and launchQDecodeStep: they write dispatches into a compute encoder that the
// caller opened and will end, commit and wait for. The whole step is one command buffer, as in
// lab 22.2 (its launch.mm is the pattern; ch02-facts C9 is why the dispatches see each other's
// writes).
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>

#include "../qdecode.h"

// Provided. Exactly n threads of `pipeline`, as lab 22.2's encodeEmbed: Metal makes the last
// threadgroup smaller, so the kernels need no guard.
static void dispatchExactly(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, uint32_t n) {
    NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (perGroup > n) perGroup = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

void encodeEmbedQ(id<MTLComputeCommandEncoder> enc, const QPipelines& p, const QMatrix& E, id<MTLBuffer> tokens,
                  id<MTLBuffer> H, uint32_t batch) {
    // TODO 3: by E.fmt, set that format's pipeline from p (embedF32, embedQ8_0, embedQ4_0), bind
    //         the buffers at the indices its kernel in kernel.metal declares and d = E.cols as
    //         a scalar (setBytes), then dispatchExactly batch * d threads. Nothing to dispatch
    //         when that is 0.
}

void encodeQLinear(id<MTLComputeCommandEncoder> enc, const QPipelines& p, id<MTLBuffer> X, const QMatrix& W,
                   id<MTLBuffer> Y, uint32_t batch) {
    // TODO 4: as launchQLinear: encodeGemvGather, encodeGemvQ8_0 or encodeGemvQ4_0 with
    //         p.gemvF32, p.gemvQ8_0 or p.gemvQ4_0 (qdecode.h has their argument orders).
}

bool encodeQDecodeStep(id<MTLComputeCommandEncoder> enc, const QDecodeModel& m, DecodeState& s) {
    // TODO 5: the step of launchQDecodeStep in kernel.cu, as dispatches into `enc`. The
    //         pipelines are in m.pipelines; decode.h declares lab 2.3's and lab 10.3's encoders.
    //         Two of those return false when they refuse to dispatch: return false then.
    return true;
}
