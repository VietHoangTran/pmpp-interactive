// Lab 22.2 starter: the Metal host side. Not in the book.
//
// encodeEmbed and encodeDecodeStep are the counterparts of launchEmbed and
// launchDecodeStep: they write dispatches into a compute encoder that the caller opened
// and will end, commit and wait for. The whole step is then one command buffer.
//
// The caller's encoder is the plain computeCommandEncoder. Before you rely on one dispatch
// seeing the previous one's writes, read notes/today/ch02-facts.md C9: what does Apple
// document about the order of dispatches in that encoder, and about buffers created from
// the device?
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>

#include "../decode.h"

void encodeEmbed(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                 id<MTLBuffer> E, id<MTLBuffer> tokens, id<MTLBuffer> H, uint32_t batch, uint32_t d) {
    // TODO 2: set the pipeline; bind E, tokens and H at indices 0, 1 and 2 and d at index 3
    //         (a scalar: setBytes); dispatch exactly batch * d threads (lab 2.3's launch.mm
    //         does this for its elementwise kernels). Nothing to dispatch when batch * d is 0.
}

bool encodeDecodeStep(id<MTLComputeCommandEncoder> enc, const DecodeModel& m, DecodeState& s) {
    // TODO 3: the step of launchDecodeStep in kernel.cu, as dispatches into `enc`. The
    //         pipelines are in m.pipelines, the GEMM is m.encodeLinear, and decode.h
    //         declares the earlier labs' encoders. Two of those return false when they
    //         refuse to dispatch: return false then.
    return true;
}
