// Lab 23.1 reference solution: the Metal host side. Not in the book: chapter 23
// is about CUDA only.
//
// On Apple silicon a shared buffer is the same memory for the CPU and the GPU, so
// section 23.3's host-to-device and device-to-host copies have no counterpart,
// and no copy engine is documented (notes/today/ch23-facts.md C4). What is left
// to hide is CPU work: writing a chunk's input into a shared buffer (the fill)
// and reading a finished chunk's output out of one (the drain). Apple documents
// one way to overlap that work with the GPU's (C5): several instances of a
// buffer in flight, a semaphore that stops the CPU from getting more than that
// many chunks ahead, and a completion handler that gives a slot back.
//
// runSerialMetal is the baseline: fill everything, run everything, drain
// everything. runInFlight is Apple's pattern with K slots.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <dispatch/dispatch.h>
#include <mach/mach_time.h>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>

#include "pmpp/metal.h"
#include "../overlap.h"

// ---- provided -------------------------------------------------------------------
// What encodePoly and waitFor saw, for the driver.
static PolyRecord g_record;

const PolyRecord& polyRecord() { return g_record; }
void polyResetRecord() {
    g_record.calls.clear();
    g_record.calls.reserve(256);
    g_record.waits = 0;
    g_record.gpuSeconds = 0;
    g_record.gpuStart.clear();
    g_record.gpuEnd.clear();
}

// Seconds of mach time, the clock GPUStartTime and GPUEndTime are given in
// (ch05-facts C9: "Both values are relative to system mach time").
static double machSeconds() {
    static mach_timebase_info_data_t base = {0, 0};
    if (base.denom == 0) mach_timebase_info(&base);
    return (double)mach_absolute_time() * base.numer / base.denom * 1e-9;
}

void encodePoly(id<MTLComputeCommandEncoder> encoder, id<MTLBuffer> inBuf, NSUInteger inOffset,
                id<MTLBuffer> outBuf, NSUInteger outOffset, int count, int degree) {
    PolyCall c{(__bridge const void*)inBuf, (__bridge const void*)outBuf, (long long)inOffset, (long long)outOffset,
               count, degree, nullptr, g_record.waits, machSeconds(), nullptr};
    const NSUInteger bytes = count > 0 ? (NSUInteger)count * sizeof(float) : 0;
    if (encoder == nil || inBuf == nil || outBuf == nil) c.refused = "a nil encoder or buffer";
    else if (count < 1) c.refused = "a count below 1";
    else if (inOffset % sizeof(float) != 0 || outOffset % sizeof(float) != 0) c.refused = "an offset that is not a multiple of 4 bytes";
    else if (inOffset + bytes > inBuf.length) c.refused = "a range past the end of the input buffer";
    else if (outOffset + bytes > outBuf.length) c.refused = "a range past the end of the output buffer";
    id<MTLComputePipelineState> pipeline = nil;
    if (c.refused == nullptr) {
        pipeline = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "poly_kernel");   // compiled once, then cached
        if (pipeline == nil) c.refused = "kernel.metal did not build";
    }
    g_record.calls.push_back(c);
    if (c.refused != nullptr) {
        printf("encodePoly: refused %s; nothing encoded\n", c.refused);
        return;
    }
    const uint32_t n = (uint32_t)count, d = (uint32_t)degree;
    [encoder setComputePipelineState:pipeline];
    [encoder setBuffer:inBuf offset:inOffset atIndex:0];
    [encoder setBuffer:outBuf offset:outOffset atIndex:1];
    [encoder setBytes:&n length:sizeof(n) atIndex:2];
    [encoder setBytes:&d length:sizeof(d) atIndex:3];
    // Whole threadgroups, as many as cover n threads; the kernel returns for
    // threads past the end.
    const NSUInteger perGroup = pipeline.maxTotalThreadsPerThreadgroup;
    [encoder dispatchThreadgroups:MTLSizeMake((n + perGroup - 1) / perGroup, 1, 1)
            threadsPerThreadgroup:MTLSizeMake(perGroup, 1, 1)];
}

int waitFor(id<MTLCommandBuffer> commandBuffer) {
    if (commandBuffer == nil) {
        printf("waitFor: nil command buffer\n");
        return 1;
    }
    if (commandBuffer.status == MTLCommandBufferStatusNotEnqueued || commandBuffer.status == MTLCommandBufferStatusEnqueued) {
        printf("waitFor: this command buffer was never committed, so it would never complete\n");
        return 1;
    }
    // Returns once the GPU has finished the command buffer and every completion
    // handler has run (infra-facts C12).
    [commandBuffer waitUntilCompleted];
    ++g_record.waits;
    if (commandBuffer.status != MTLCommandBufferStatusCompleted) {
        NSError* error = commandBuffer.error;
        printf("command buffer failed (%s)\n", error ? error.localizedDescription.UTF8String : "no error object");
        return 1;
    }
    // The GPU's own account of the command buffer, valid once it has completed (ch05-facts C9).
    const double s = commandBuffer.GPUStartTime, e = commandBuffer.GPUEndTime;
    g_record.gpuSeconds += e - s;
    g_record.gpuStart.push_back(s);
    g_record.gpuEnd.push_back(e);
    return 0;
}

// The two n-float buffers stay from one call to the next (made on the first call,
// which the benchmark does not time), as the CUDA driver keeps d_in and d_out.
// So a timed call is exactly fill, kernel, drain.
static id<MTLBuffer> g_serialIn = nil, g_serialOut = nil;

int runSerialMetal(const MetalJob& job) {
    const size_t bytes = (size_t)job.n * sizeof(float);
    if (g_serialIn == nil || g_serialIn.length < bytes) {
        g_serialIn = [job.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        g_serialOut = [job.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        if (g_serialIn == nil || g_serialOut == nil) {
            printf("buffer creation failed in %s at line %d\n", __FILE__, __LINE__);
            g_serialIn = g_serialOut = nil;
            return 1;
        }
    }
    memcpy(g_serialIn.contents, job.hostIn, bytes);   // the fill
    id<MTLCommandBuffer> commandBuffer = [job.queue commandBuffer];
    id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
    if (commandBuffer == nil || encoder == nil) {
        printf("no command buffer or encoder in %s at line %d\n", __FILE__, __LINE__);
        return 1;
    }
    encodePoly(encoder, g_serialIn, 0, g_serialOut, 0, job.n, job.degree);
    [encoder endEncoding];
    [commandBuffer commit];
    if (waitFor(commandBuffer) != 0) return 1;
    memcpy(job.hostOut, g_serialOut.contents, bytes);  // the drain, after the wait (ch05-facts C6)
    return 0;
}

// The slot pool. A new buffer is cheap to create, but on the machine this lab
// was built on, the first write to each of its pages took several times as long
// as a write to a buffer already used (README.md; the benchmark's firsttouch row). Keeping the buffers between calls
// keeps that cost out of every call but the first, which the benchmark does not
// time. Your code decides how many slots and how long they are.
static const int kMaxSlots = 64;
static id<MTLBuffer> g_slots[kMaxSlots][2];

id<MTLBuffer> slotBuffer(id<MTLDevice> device, int slot, int which, NSUInteger bytes) {
    if (slot < 0 || slot >= kMaxSlots || (which != 0 && which != 1) || bytes == 0) {
        printf("slotBuffer: slot %d, which %d, %lu bytes: slot must be 0 to %d, which 0 or 1, bytes above 0\n", slot,
               which, (unsigned long)bytes, kMaxSlots - 1);
        return nil;
    }
    id<MTLBuffer> __strong& b = g_slots[slot][which];
    if (b == nil || b.length < bytes) b = [device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if (b == nil) printf("slotBuffer: newBufferWithLength of %lu bytes returned nil\n", (unsigned long)bytes);
    return b;
}

// ---- your code below ------------------------------------------------------------

// One instance of the buffers: the chunk it holds, and the command buffer that
// computes it (nil when the slot is free or already drained).
struct Slot {
    id<MTLBuffer> in;
    id<MTLBuffer> out;
    id<MTLCommandBuffer> pending;
    int begin;
    int count;
};

// Waits for the slot's chunk and copies its output to hostOut. On the calling
// thread, which is the main thread: the completion handler does not do this,
// because the facts files record when a handler runs, not on which thread.
static int drain(const MetalJob& job, Slot& s) {
    if (s.pending == nil) return 0;
    const int rc = waitFor(s.pending);
    if (rc == 0) memcpy(job.hostOut + s.begin, s.out.contents, (size_t)s.count * sizeof(float));
    s.pending = nil;
    return rc;
}

int runInFlight(const MetalJob& job, int buffers, int chunks) {
    const int K = buffers;
    // The largest chunk: ceil(n / chunks) elements. No chunk of the rule is longer.
    const size_t slotBytes = (size_t)((job.n + chunks - 1) / chunks) * sizeof(float);
    std::vector<Slot> slot((size_t)K);
    for (int i = 0; i < K; ++i) {
        Slot& s = slot[(size_t)i];
        s.in = slotBuffer(job.device, i, 0, slotBytes);    // shared storage: the CPU fills and
        s.out = slotBuffer(job.device, i, 1, slotBytes);   // drains them in place
        s.pending = nil;
        s.begin = s.count = 0;
        if (s.in == nil || s.out == nil) return 1;          // slotBuffer printed why
    }

    // K tokens, one per slot (C5). A token is taken before a chunk is filled and
    // given back by its command buffer's completion handler.
    dispatch_semaphore_t freeSlots = dispatch_semaphore_create(K);
    int rc = 0;
    int j = 0;   // chunks queued so far
    for (int k = 0; k < chunks && rc == 0; ++k) {
        const int begin = (int)((long long)k * job.n / chunks);
        const int end = (int)((long long)(k + 1) * job.n / chunks);
        if (end == begin) continue;

        dispatch_semaphore_wait(freeSlots, DISPATCH_TIME_FOREVER);
        Slot& s = slot[(size_t)(j % K)];
        // The token says some command buffer finished; waitFor makes sure it was
        // this slot's, and reports a failure if the GPU did not run it.
        if (drain(job, s) != 0) { rc = 1; dispatch_semaphore_signal(freeSlots); break; }

        // Every CPU write to the buffer is done before the commit that uses it (C5).
        memcpy(s.in.contents, job.hostIn + begin, (size_t)(end - begin) * sizeof(float));
        id<MTLCommandBuffer> commandBuffer = [job.queue commandBuffer];
        id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
        if (commandBuffer == nil || encoder == nil) {
            printf("no command buffer or encoder in %s at line %d\n", __FILE__, __LINE__);
            rc = 1;
            dispatch_semaphore_signal(freeSlots);   // this token was taken but will not come back
            break;
        }
        encodePoly(encoder, s.in, 0, s.out, 0, end - begin, job.degree);
        [encoder endEncoding];
        // The handler only gives the token back. It captures the semaphore, which
        // keeps it alive until the last handler has run.
        [commandBuffer addCompletedHandler:^(id<MTLCommandBuffer>) {
            dispatch_semaphore_signal(freeSlots);
        }];
        [commandBuffer commit];
        s.pending = commandBuffer;
        s.begin = begin;
        s.count = end - begin;
        ++j;
    }

    // The chunks still in flight, oldest first. Also on the error path: every
    // committed command buffer must finish, so that every token comes back. On the
    // machine this lab was built on, releasing a dispatch semaphore whose count is
    // below its starting value ended the process.
    for (int i = 0; i < K; ++i)
        if (drain(job, slot[(size_t)((j + i) % K)]) != 0) rc = 1;
    return rc;
}
