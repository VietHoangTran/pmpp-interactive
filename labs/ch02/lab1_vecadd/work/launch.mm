// Lab 2.1 starter: the Metal host side. Not in the book.
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

// Contract: returns 0 when everything worked, non-zero otherwise. Metal has no
// error codes: a creation method that fails returns nil, and a failure of the
// kernel shows on the command buffer after the wait. Check each of them.
int vecAdd(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
           const float* A_h, const float* B_h, float* C_h, uint32_t n) {
    // TODO 1: Part 1. Three buffers with shared storage. The two inputs are
    //         created from the host arrays in one call each. The output is
    //         created by its length. Lengths are in bytes.

    // TODO 2: Part 2. A command buffer from the queue and a compute encoder
    //         from the command buffer. Give the encoder the pipeline and the
    //         three buffers, at the indices kernel.metal declares.

    // TODO 3: Dispatch exactly n threads (threads, not threadgroups). The
    //         threadgroup size comes from the pipeline and must not be larger
    //         than the grid. Then end the encoding.

    // TODO 4: Submit the command buffer and wait for it. Only then can its
    //         status and error tell you whether the kernel ran.

    // TODO 5: Part 3. Copy the result from the output buffer to C_h.
    //         What replaces cudaFree?

    return -1;   // replace with 0 once the function is complete
}
