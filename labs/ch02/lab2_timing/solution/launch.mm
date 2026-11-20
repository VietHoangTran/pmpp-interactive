// Lab 2.2 reference solution: the Metal host side. Not in the book.
// The vecAdd of lab 2.1 with a clock read between its parts, and the same
// launch on data that already lives in shared buffers.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Provided: a clock that never runs backwards, and the seconds between two reads.
typedef std::chrono::steady_clock::time_point Instant;
static Instant now() { return std::chrono::steady_clock::now(); }
static double seconds(Instant from, Instant to) {
    return std::chrono::duration<double>(to - from).count();
}

// Encodes one dispatch of vec_add over exactly n threads, commits it and
// returns the command buffer without waiting. Returns nil on failure.
static id<MTLCommandBuffer> submitVecAdd(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                                         id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint32_t n) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    if (commandBuffer == nil) return nil;
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (encoder == nil) return nil;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:A offset:0 atIndex:0];
    [encoder setBuffer:B offset:0 atIndex:1];
    [encoder setBuffer:C offset:0 atIndex:2];
    NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > n) threadsPerGroup = n;
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [encoder endEncoding];
    [commandBuffer commit];
    return commandBuffer;
}

// Waits for a committed command buffer and says whether the GPU ran it.
static int finish(id<MTLCommandBuffer> commandBuffer, int line) {
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s) in %s at line %d\n",
               error ? error.localizedDescription.UTF8String : "no error object", __FILE__, line);
        return 1;
    }
    return 0;
}

// phase_s[0..4]: seconds spent creating the buffers and copying the host
// arrays into them, encoding and committing, waiting, copying the result out;
// and, last, the GPU time the command buffer reports.
int vecAddTimed(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                const float* A_h, const float* B_h, float* C_h, uint32_t n, double* phase_s) {
    const NSUInteger size = (NSUInteger)n * sizeof(float);

    // Phase 0: buffers. newBufferWithBytes allocates and copies in one call, so
    // allocation and copy-in cannot be told apart from outside, and both are
    // finished when the call returns: nothing here is queued for later.
    Instant t0 = now();
    id<MTLBuffer> A = [device newBufferWithBytes:A_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [device newBufferWithBytes:B_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (A == nil || B == nil || C == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    Instant t1 = now();

    // Phase 1: encode and commit. commit returns before the GPU has run
    // anything, so this phase is host work only.
    id<MTLCommandBuffer> commandBuffer = submitVecAdd(queue, pipeline, A, B, C, n);
    if (commandBuffer == nil) {
        printf("no command buffer or encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    Instant t2 = now();

    // Phase 2: the wait. Everything between the commit and the result being
    // ready is in here: scheduling, the kernel, and waking this thread up.
    if (finish(commandBuffer, __LINE__) != 0) return 1;
    Instant t3 = now();

    // Phase 3: copy out. There is no version of this function without the wait
    // above: reading contents while the GPU may still be writing the buffer is
    // an ordering error, and the copy would not wait for the kernel as a
    // blocking cudaMemcpy does.
    memcpy(C_h, C.contents, size);
    Instant t4 = now();

    phase_s[0] = seconds(t0, t1);
    phase_s[1] = seconds(t1, t2);
    phase_s[2] = seconds(t2, t3);
    phase_s[3] = seconds(t3, t4);
    // The command buffer's own account of when the GPU started and finished
    // it. Both times are 0 until the wait has returned.
    phase_s[4] = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    return 0;
    // The three buffers are released after this point, when their last
    // references go away. No clock in this function can see that.
}

// The same launch on buffers the caller owns: created with shared storage and
// filled through contents. Part 2 of vecAdd is all that is left. The caller
// reads the result in place, through C.contents, after this returns 0.
int vecAddResident(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint32_t n) {
    id<MTLCommandBuffer> commandBuffer = submitVecAdd(queue, pipeline, A, B, C, n);
    if (commandBuffer == nil) {
        printf("no command buffer or encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    return finish(commandBuffer, __LINE__);
}
