// Lab 2.1 reference solution: the Metal host side. Not in the book.
// The same three parts as the book's Fig. 2.13, in Metal's vocabulary.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstring>

// Provided. Turns the text of kernel.metal into something the GPU can run:
// source -> library -> function -> pipeline state. In CUDA this is the work of
// nvcc and the driver; here the program does it at run time, once, and the
// driver of this lab keeps the result. Returns nil on failure, with *error set
// by the two steps that compile.
id<MTLComputePipelineState> makeVecAddPipeline(id<MTLDevice> device, NSString* mslSource, NSError** error) {
    id<MTLLibrary> library = [device newLibraryWithSource:mslSource options:nil error:error];
    if (library == nil) return nil;                 // a compile error; *error has the message

    id<MTLFunction> function = [library newFunctionWithName:@"vec_add"];
    if (function == nil) return nil;                // no kernel of that name; *error is not set

    // The function object is not executable code yet. The pipeline state is.
    return [device newComputePipelineStateWithFunction:function error:error];
}

int vecAdd(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
           const float* A_h, const float* B_h, float* C_h, uint32_t n) {
    const NSUInteger size = (NSUInteger)n * sizeof(float);    // bytes, not elements

    // Part 1: buffers. There is no error code: a failed creation returns nil.
    // newBufferWithBytes allocates and copies the host array in one call (the
    // counterpart of cudaMalloc + cudaMemcpy). newBufferWithLength allocates
    // only, and clears the memory to zero.
    id<MTLBuffer> A = [device newBufferWithBytes:A_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> B = [device newBufferWithBytes:B_h length:size options:MTLResourceStorageModeShared];
    id<MTLBuffer> C = [device newBufferWithLength:size options:MTLResourceStorageModeShared];
    if (A == nil || B == nil || C == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }

    // Part 2: the launch. One CUDA statement becomes a command buffer, an
    // encoder, the kernel, its arguments and a dispatch.
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

    // dispatchThreads takes the number of threads, not of threadgroups. The
    // grid is exactly n, and Metal makes the last threadgroup smaller when n is
    // not a multiple of the group size: no ceiling division and no guard.
    NSUInteger threadsPerGroup = pipeline.maxTotalThreadsPerThreadgroup;
    if (threadsPerGroup > n) threadsPerGroup = n;
    [encoder dispatchThreads:MTLSizeMake(n, 1, 1) threadsPerThreadgroup:MTLSizeMake(threadsPerGroup, 1, 1)];
    [encoder endEncoding];

    [commandBuffer commit];                // returns at once, like a CUDA launch
    [commandBuffer waitUntilCompleted];    // the counterpart of cudaDeviceSynchronize()

    // A failure of the kernel itself is visible only now.
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s) in %s at line %d\n",
               error ? error.localizedDescription.UTF8String : "no error object", __FILE__, __LINE__);
        return 1;
    }

    // Part 3: a shared buffer is ordinary memory to the CPU once the GPU has
    // finished with it. Nothing to free: the buffers go away with their last
    // reference, when this function returns.
    memcpy(C_h, C.contents, size);
    return 0;
}
