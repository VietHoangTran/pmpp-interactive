// Lab 2.3 reference solution: the Metal host side. Not in the book.
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

// One thread per element: the grid is exactly n threads, and Metal makes the
// last threadgroup smaller when n is not a multiple of the group size. No
// ceiling division, and no guard in the kernel.
static void dispatchElements(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline, uint32_t n) {
    if (n == 0) return;                               // an empty tensor: nothing to dispatch
    NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > n) threadsPerGroup = n;
    [enc dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
}

void encodeAddBias(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> Y, id<MTLBuffer> X, id<MTLBuffer> bias, uint32_t rows, uint32_t d) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:Y offset:0 atIndex:0];
    [enc setBuffer:X offset:0 atIndex:1];
    [enc setBuffer:bias offset:0 atIndex:2];
    [enc setBytes:&d length:sizeof(d) atIndex:3];     // a scalar: copied into the argument table
    dispatchElements(enc, pipeline, rows * d);
}

void encodeScaleShift(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> Y, id<MTLBuffer> X, float alpha, float beta, uint32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:Y offset:0 atIndex:0];
    [enc setBuffer:X offset:0 atIndex:1];
    [enc setBytes:&alpha length:sizeof(alpha) atIndex:2];
    [enc setBytes:&beta length:sizeof(beta) atIndex:3];
    dispatchElements(enc, pipeline, n);
}

void encodeAddResidual(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> X, id<MTLBuffer> R, uint32_t n) {
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:X offset:0 atIndex:0];
    [enc setBuffer:R offset:0 atIndex:1];
    dispatchElements(enc, pipeline, n);
}

// y = alpha * (x + bias) + beta + r, for host arrays, with one command buffer.
int blockTail(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> addBias, id<MTLComputePipelineState> scaleShift,
              id<MTLComputePipelineState> addResidual,
              const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
              float* y_h, uint32_t rows, uint32_t d) {
    const uint32_t n = rows * d;
    const NSUInteger size = (NSUInteger)n * sizeof(float);         // bytes of one tensor
    const NSUInteger biasSize = (NSUInteger)d * sizeof(float);     // bytes of one row

    // In. The interface hands us ordinary host arrays, so each one is copied
    // into a shared buffer once. (Data that lived in shared buffers from the
    // start would not be copied at all.)
    id<MTLBuffer> X = [device newBufferWithBytes:x_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [device newBufferWithBytes:bias_h length:biasSize options:MTLResourceStorageModeShared];
    id<MTLBuffer> R = [device newBufferWithBytes:r_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> Y = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (X == nil || B == nil || R == nil || Y == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }

    // One command buffer, one encoder, three dispatches. Each kernel must see
    // what the one before it wrote, and Apple documents two reasons it does
    // (notes/today/ch02-facts.md, C9). The plain computeCommandEncoder
    // "dispatches its compute commands serially"; a concurrent encoder has to
    // be asked for. And buffers created from the device are hazard-tracked by
    // default, so Metal will "prevent subsequent commands from running until
    // write operations finish". An encoder made with MTLDispatchTypeConcurrent,
    // or buffers from a heap, would need a barrier between these dispatches.
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) {
        printf("no command buffer in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) {
        printf("no compute encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    encodeAddBias(encoder, addBias, Y, X, B, rows, d);            // y = x + bias
    encodeScaleShift(encoder, scaleShift, Y, Y, alpha, beta, n);  // y = alpha * y + beta (a thread reads and writes only its own element)
    encodeAddResidual(encoder, addResidual, Y, R, n);             // y = y + r
    [encoder endEncoding];

    // One commit and one wait for all three.
    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s) in %s at line %d\n",
               error ? error.localizedDescription.UTF8String : "no error object", __FILE__, __LINE__);
        return 1;
    }

    // Out: the one result, once. The buffers go away with their last reference.
    memcpy(y_h, Y.contents, size);
    return 0;
}
