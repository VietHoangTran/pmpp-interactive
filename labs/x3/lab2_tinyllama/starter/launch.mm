// Lab X3.2 starter: the Metal host side. Not in the book.
//
// encodeLlamaStep is the counterpart of launchLlamaStep: it writes every dispatch of one
// step into the compute encoder the caller opened and will end and commit, so one step is
// one command buffer.
//
// The caller's encoder is the plain computeCommandEncoder. Before you rely on one dispatch
// seeing the previous one's writes, read notes/today/ch02-facts.md C9: what does Apple
// document about the order of dispatches in that encoder, and about buffers made by the
// device?
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cmath>
#include <cstdint>

#include "../llama.h"

void encodeAdvancePositions(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                            const LlamaState& s) {
    // TODO 2: set the pipeline; bind tokens, pos, len, prompt, out and rngStep at indices 0 to
    //         5, promptLen and batch at 6 and 7 (scalars: setBytes); dispatch exactly s.batch
    //         threads (lab 22.2's encodeEmbed dispatches an exact count).
}

bool encodeLlamaStep(id<MTLComputeCommandEncoder> enc, const LlamaModel& m, LlamaState& s) {
    // TODO 3: the step of launchLlamaStep in kernel.cu, as dispatches into `enc`. The pipelines
    //         are in m.pipelines, the GEMM is m.encodeLinear, the attention m.encodeAttention,
    //         the sampler s.encodeSample (null for greedy), and llama.h declares the earlier
    //         labs' encoders. encodeRmsNorm, encodeArgmaxRows and the sampler return false
    //         when they refuse to dispatch: return false then.
    return true;
}
