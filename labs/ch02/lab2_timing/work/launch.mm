// Lab 2.2 starter: the Metal host side. Not in the book.
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

// Below is the reference vecAdd of lab 2.1 under a new name (paste your own if
// you prefer it). It computes the right sums and measures nothing.
//
// Contract:
//   - phase_s[0..3] receive the seconds spent in: creating the buffers and
//     copying the host arrays into them; encoding and committing; waiting;
//     copying the result out. Use now() and seconds() above.
//   - phase_s[4] receives the GPU time the command buffer itself reports
//     (GPUEndTime - GPUStartTime, in seconds).
//   - returns 0, or non-zero when something failed, as in lab 2.1.
// There is no syncAfterLaunch argument here. On CUDA the copy back waits for
// the kernel by itself, so leaving the synchronise out only moves time from one
// phase to another. Here nothing waits for you: reading contents before the
// wait has returned is an ordering error, not a timing choice.
int vecAddTimed(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                const float* A_h, const float* B_h, float* C_h, uint32_t n, double* phase_s) {
    const NSUInteger size = (NSUInteger)n * sizeof(float);

    // TODO 1: decide where each clock read goes. Four phases need five reads.

    id<MTLBuffer> A = [device newBufferWithBytes:A_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [device newBufferWithBytes:B_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (A == nil || B == nil || C == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }

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
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:A offset:0 atIndex:0];
    [encoder setBuffer:B offset:0 atIndex:1];
    [encoder setBuffer:C offset:0 atIndex:2];
    NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > n) threadsPerGroup = n;
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [encoder endEncoding];

    [commandBuffer commit];
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s) in %s at line %d\n",
               error ? error.localizedDescription.UTF8String : "no error object", __FILE__, __LINE__);
        return 1;
    }

    memcpy(C_h, C.contents, size);

    // TODO 2: fill phase_s[0..3], in seconds.

    // TODO 3: fill phase_s[4] from the command buffer. When does it know the
    //         answer?

    // TODO 4: the three buffers are released when this function returns. Which
    //         of your phases contains that? (Nothing to write. The benchmark
    //         prints the driver's time for the whole call beside your total.)
    return 0;
}

// TODO 5: the same launch on buffers the caller owns. The driver created A, B
//         and C with shared storage and wrote the inputs through contents; it
//         reads the result in place, through C.contents, after you return 0.
//         Which lines of vecAddTimed are still needed? Checks as before.
int vecAddResident(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint32_t n) {
    return -1;   // replace with 0 once the function is complete
}
