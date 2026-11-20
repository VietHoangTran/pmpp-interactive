// Lab 2.3 starter: the Metal host side. Not in the book.
//
// The counterpart of a launch<Name> function is an encode<Name> function: it
// writes one dispatch into a compute encoder that somebody else opened and
// somebody else will close, commit and wait for. That is what lets the caller
// put several dispatches in one command buffer.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// TODO 5: the three encode functions. Each one gives the encoder its pipeline,
//         binds the buffers at the indices kernel.metal declares, sends the
//         scalars the kernel takes (they are not buffers), and dispatches one
//         thread per element, as you did in lab 2.1. Nothing else: the encoder
//         is not theirs to end, and there is nothing to commit or wait for.
void encodeAddBias(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> Y, id<MTLBuffer> X, id<MTLBuffer> bias, uint32_t rows, uint32_t d) {
}

void encodeScaleShift(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> Y, id<MTLBuffer> X, float alpha, float beta, uint32_t n) {
}

void encodeAddResidual(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> X, id<MTLBuffer> R, uint32_t n) {
}

// y = alpha * (x + bias) + beta + r, for host arrays, with one command buffer.
// Returns 0 when everything worked and a non-zero value otherwise.
int blockTail(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> addBias, id<MTLComputePipelineState> scaleShift,
              id<MTLComputePipelineState> addResidual,
              const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
              float* y_h, uint32_t rows, uint32_t d) {
    // TODO 6: buffers with shared storage for the three inputs and the result.
    //         The interface hands you host arrays, so each input costs one
    //         copy. The bias is one row, not a whole tensor.

    // TODO 7: one command buffer and one compute encoder. Call your three
    //         encode functions on it, in order. Which buffer does each one
    //         read, and which does it write? Then end the encoding.

    // TODO 8: commit once and wait once. Then check how the command buffer
    //         ended.

    // TODO 9: copy the one result to y_h.

    return -1;   // replace with 0 once the function is complete
}
