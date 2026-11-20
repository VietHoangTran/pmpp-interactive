// Lab C.1 reference solution: the Metal host side. Not in the book: appendix C is
// about CUDA only. On Apple silicon the CPU and the GPU share one memory
// (ch05-facts C6), so the three paths here are not the book's three. What they
// differ in is whether the table is copied, and who owns the memory:
//   feedPrivateCopy  the book's explicit model: a staging buffer, a blit into a
//                    private buffer that only the GPU can touch, and a blit back;
//   feedNoCopy       memory the program already owns, wrapped in place: what
//                    llama.cpp does with memory-mapped weights (appC-facts D4);
//   feedShared       a buffer Metal allocates, which both sides use (Apple's
//                    advice for data the CPU fills and the GPU reads, appC-facts C2).
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

// Encodes every pass into one compute pass, commits it and waits. The dispatches
// of a plain compute encoder run in order and each sees the previous one's
// writes (ch02-facts C9), which the += in the kernel relies on.
static int runPasses(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline, id<MTLBuffer> table,
                     id<MTLBuffer> ids, id<MTLBuffer> out, const FeedJob* job) {
    id<MTLCommandBuffer> commandBuffer = [queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        printf("no command buffer or encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    for (int p = 0; p < job->passes; ++p)
        encodeGatherAdd(encoder, pipeline, table, ids, out, (uint32_t)job->count, (uint32_t)job->cols);
    [encoder endEncoding];
    [commandBuffer commit];
    return waitFor(commandBuffer, kKernelWork);
}

// 1. The explicit model: the CPU fills a staging buffer, a blit copies it into
//    memory only the GPU can touch, the kernels run there, a blit copies the
//    result back. Each step is its own command buffer, waited for before the
//    next, so the GPU reports copies and kernels separately.
int feedPrivateCopy(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                    const FeedJob* job) {
    const size_t tableBytes = feedTableBytes(job), idsBytes = feedIdsBytes(job), outBytes = feedOutBytes(job);

    // A private buffer has no CPU address (contents is NULL, ch02-facts C2), so the
    // CPU writes into shared staging buffers first (ch02-facts C3).
    id<MTLBuffer> stageTable = [device newBufferWithLength:tableBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> stageIds = [device newBufferWithBytes:job->ids_h length:idsBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> table = [device newBufferWithLength:tableBytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> ids = [device newBufferWithLength:idsBytes options:MTLResourceStorageModePrivate];
    // newBufferWithLength clears the buffer with zeros (ch02-facts C2): out starts at zero.
    id<MTLBuffer> out = [device newBufferWithLength:outBytes options:MTLResourceStorageModePrivate];
    id<MTLBuffer> result = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];
    if (stageTable == nil || stageIds == nil || table == nil || ids == nil || out == nil || result == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    job->produce((float*)stageTable.contents, job->ctx);

    // Copy in. Sizes are whole floats and ints, so multiples of 4 bytes, as a
    // blit on macOS requires (ch02-facts C3).
    id<MTLCommandBuffer> copyIn = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitIn = [copyIn blitCommandEncoder];
    if (copyIn == nil || blitIn == nil) {
        printf("no command buffer or blit encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    [blitIn copyFromBuffer:stageTable sourceOffset:0 toBuffer:table destinationOffset:0 size:tableBytes];
    [blitIn copyFromBuffer:stageIds sourceOffset:0 toBuffer:ids destinationOffset:0 size:idsBytes];
    [blitIn endEncoding];
    [copyIn commit];
    if (waitFor(copyIn, kCopyWork) != 0) return 1;

    if (runPasses(queue, pipeline, table, ids, out, job) != 0) return 1;

    // Copy out, into a buffer the CPU can read.
    id<MTLCommandBuffer> copyOut = [queue commandBuffer];
    id<MTLBlitCommandEncoder> blitOut = [copyOut blitCommandEncoder];
    if (copyOut == nil || blitOut == nil) {
        printf("no command buffer or blit encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    [blitOut copyFromBuffer:out sourceOffset:0 toBuffer:result destinationOffset:0 size:outBytes];
    [blitOut endEncoding];
    [copyOut commit];
    if (waitFor(copyOut, kCopyWork) != 0) return 1;

    job->consume((const float*)result.contents, job->ctx);
    return 0;   // the buffers are released when the last reference to each goes away
}

// Page-aligned memory of a whole number of pages: what newBufferWithBytesNoCopy
// asks for (ch02-facts C2). `length` receives the rounded size.
static void* pageAlloc(size_t bytes, size_t* length) {
    const size_t page = (size_t)getpagesize();
    *length = (bytes + page - 1) / page * page;
    void* p = nullptr;
    if (posix_memalign(&p, page, *length) != 0) return nullptr;
    return p;
}

// Wraps memory this program allocated; the deallocator frees it when Metal
// releases the buffer. If wrapping fails, the memory is freed here instead.
static id<MTLBuffer> wrap(id<MTLDevice> device, void* p, size_t length) {
    if (p == nullptr) return nil;
    id<MTLBuffer> b = [device newBufferWithBytesNoCopy:p
                                                length:length
                                               options:MTLResourceStorageModeShared
                                           deallocator:^(void* pointer, NSUInteger) { free(pointer); }];
    if (b == nil) free(p);
    return b;
}

// 2. Memory the program already owns, used by the GPU in place.
int feedNoCopy(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
               const FeedJob* job) {
    const size_t tableBytes = feedTableBytes(job), idsBytes = feedIdsBytes(job), outBytes = feedOutBytes(job);

    size_t tableLength, idsLength, outLength;
    float* table_h = (float*)pageAlloc(tableBytes, &tableLength);
    int* ids_h = (int*)pageAlloc(idsBytes, &idsLength);
    float* out_h = (float*)pageAlloc(outBytes, &outLength);
    id<MTLBuffer> table = wrap(device, table_h, tableLength);
    id<MTLBuffer> ids = wrap(device, ids_h, idsLength);
    id<MTLBuffer> out = wrap(device, out_h, outLength);
    if (table == nil || ids == nil || out == nil) {
        printf("page-aligned allocation or newBufferWithBytesNoCopy failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;   // the buffers that did get made free their memory when released
    }

    job->produce(table_h, job->ctx);   // the CPU writes the memory the GPU will read
    memcpy(ids_h, job->ids_h, idsBytes);
    memset(out_h, 0, outBytes);         // posix_memalign memory is not cleared

    if (runPasses(queue, pipeline, table, ids, out, job) != 0) return 1;
    job->consume(out_h, job->ctx);      // after the wait: the GPU has finished writing
    return 0;
}

// 3. A buffer Metal allocates, used by both sides in place.
int feedShared(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
               const FeedJob* job) {
    const size_t tableBytes = feedTableBytes(job), idsBytes = feedIdsBytes(job), outBytes = feedOutBytes(job);

    id<MTLBuffer> table = [device newBufferWithLength:tableBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> ids = [device newBufferWithBytes:job->ids_h length:idsBytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> out = [device newBufferWithLength:outBytes options:MTLResourceStorageModeShared];   // zeros (C2)
    if (table == nil || ids == nil || out == nil) {
        printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    job->produce((float*)table.contents, job->ctx);

    if (runPasses(queue, pipeline, table, ids, out, job) != 0) return 1;
    job->consume((const float*)out.contents, job->ctx);   // after the wait (ch05-facts C6)
    return 0;
}
