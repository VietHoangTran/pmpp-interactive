// Lab C.1 starter: the Metal host side. Not in the book: appendix C is about CUDA
// only. On Apple silicon the CPU and the GPU share one memory (ch05-facts C6), so
// the three paths here are not the book's three. What they differ in is whether
// the table is copied, and who owns the memory:
//   feedPrivateCopy  the book's explicit model: a staging buffer, a blit into a
//                    private buffer that only the GPU can touch, and a blit back;
//   feedNoCopy       memory the program already owns, wrapped in place: what
//                    llama.cpp does with memory-mapped weights (appC-facts D4);
//   feedShared       a buffer Metal allocates, which both sides use (Apple's
//                    advice for data the CPU fills and the GPU reads, appC-facts C2).
//
// kernel.metal is provided complete. The contract for each path (feeding.h):
//   - make the table, the ids and the result live in this path's memory;
//   - call job->produce(table, job->ctx) exactly once to write the table, with a
//     pointer the CPU may write through;
//   - give the ids the values of job->ids_h, and start the result at zero;
//   - encode job->passes calls of encodeGatherAdd, commit, and wait with waitFor
//     (kKernelWork for kernels, kCopyWork for blits), which also reports failures;
//   - call job->consume(out, job->ctx) exactly once, with a pointer the CPU may read;
//   - return 0, or 1 after printing what failed (a nil buffer, a failed wait).
// Buffers are Objective-C objects: they go away with their last reference.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>

#include "../feeding.h"

// ---- provided -------------------------------------------------------------------
// What the launcher was last given, and GPU time so far, for the driver.
static const void* g_lastTable = nullptr;
static int g_tablePrivate = -1;
static int g_launches = 0;
static double g_copy_s = 0, g_kernel_s = 0;

void encodeGatherAdd(id<MTLComputeCommandEncoder> encoder, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> table, id<MTLBuffer> ids, id<MTLBuffer> out, uint32_t count, uint32_t cols) {
    g_lastTable = table.contents;   // NULL for a private buffer (ch02-facts C2)
    g_tablePrivate = table.storageMode == MTLStorageModePrivate ? 1 : 0;
    ++g_launches;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:table offset:0 atIndex:0];
    [encoder setBuffer:ids offset:0 atIndex:1];
    [encoder setBuffer:out offset:0 atIndex:2];
    [encoder setBytes:&count length:sizeof(count) atIndex:3];
    [encoder setBytes:&cols length:sizeof(cols) atIndex:4];
    // Whole threadgroups, as many as cover count * cols threads; the kernel
    // returns for threads past the end.
    const NSUInteger n = (NSUInteger)count * cols;
    const NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake((n + perGroup - 1) / perGroup, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

int waitFor(id<MTLCommandBuffer> commandBuffer, FeedWork kind) {
    [commandBuffer waitUntilCompleted];
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s)\n", error ? error.localizedDescription.UTF8String : "no error object");
        return 1;
    }
    // The GPU's own account of the command buffer, valid once it has completed (ch05-facts C9).
    const double s = commandBuffer.GPUEndTime - commandBuffer.GPUStartTime;
    if (kind == kCopyWork) g_copy_s += s;
    else g_kernel_s += s;
    return 0;
}

FeedRecord feedRecord() {
    FeedRecord r;
    r.table = g_lastTable;
    r.launches = g_launches;
    r.tablePrivate = g_tablePrivate;
    r.gpu_copy_s = g_copy_s;
    r.gpu_kernel_s = g_kernel_s;
    return r;
}
void feedResetRecord() {
    g_lastTable = nullptr;
    g_tablePrivate = -1;
    g_launches = 0;
    g_copy_s = g_kernel_s = 0;
}

// ---- your code below ------------------------------------------------------------

// 1. The explicit model: the CPU fills a staging buffer, a copy puts the data
//    where only the GPU can reach it, the kernels run there, a copy brings the
//    result back.
int feedPrivateCopy(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                    const FeedJob* job) {
    // TODO 1: the GPU's copies of the table, the ids and the result: private
    //         storage. What does `contents` return for such a buffer
    //         (ch02-facts C2)? So where must produce write?
    // TODO 2: copy in with a blit encoder, in a command buffer of its own, so the
    //         GPU reports the copy apart from the kernels. Wait with kCopyWork.
    // TODO 3: the passes, in one compute encoder, in another command buffer.
    //         Wait with kKernelWork.
    // TODO 4: copy the result into a buffer the CPU can read, wait, consume.
    (void)device; (void)queue; (void)pipeline; (void)job;
    printf("feedPrivateCopy: not written yet\n");
    return -1;
}

// 2. Memory the program already owns, used by the GPU in place.
int feedNoCopy(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
               const FeedJob* job) {
    // TODO 1: allocate host memory that newBufferWithBytesNoCopy accepts. Read
    //         ch02-facts C2 for its two conditions; getpagesize() is in <unistd.h>.
    // TODO 2: wrap each allocation in a buffer. Decide who frees the memory, and
    //         when it is safe to: the deallocator argument is one answer.
    // TODO 3: produce, ids, a zero result, written by the CPU into the memory you
    //         allocated.
    // TODO 4: the passes, the wait, consume.
    (void)device; (void)queue; (void)pipeline; (void)job;
    printf("feedNoCopy: not written yet\n");
    return -1;
}

// 3. A buffer Metal allocates, used by both sides in place.
int feedShared(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
               const FeedJob* job) {
    // TODO 1: three buffers with shared storage. Which creation call fills a
    //         buffer with zeros for you (ch02-facts C2)?
    // TODO 2: produce into the table, through the buffer's CPU address.
    // TODO 3: the passes, the wait, consume. Not one byte is copied by Metal.
    (void)device; (void)queue; (void)pipeline; (void)job;
    printf("feedShared: not written yet\n");
    return -1;
}
