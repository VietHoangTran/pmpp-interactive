// Lab 18.1 (written for you; you need not edit this file): the Metal host loop for
// bfs_privatised, the counterpart of launchBfsPrivatised in kernel.cu, with the same contract.
//
// Metal has no grid-wide barrier, no cooperative launch and no occupancy query
// (notes/today/ch18-facts.md C1), so every level is its own command buffer: zero the counter,
// encode one dispatch, commit, wait, read the count the GPU wrote, decide whether to go on.
// The buffers use shared storage (the lab runtime's choice), so the CPU writes the root and
// zeroes the counter directly between command buffers. Dispatches on one MTLCommandQueue with
// default (tracked) buffers run in order and see each other's writes (C2).
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#ifndef BLOCK_DIM
#define BLOCK_DIM 128
#endif

// Before the call: level[v] = UNVISITED for every v except level[root] = 0, and frontierA and
// frontierB each have room for numVertices entries. Returns the number of dispatches; the last
// one finds no new vertex. If frontierSizes is not null (numVertices + 1 entries), it receives
// each level's frontier size, frontierSizes[0] = 1 for the root. A GPU error stops the loop and
// is described in *error.
unsigned int launchBfsPrivatised(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                                 id<MTLBuffer> srcPtrs, id<MTLBuffer> dst, uint32_t numVertices, uint32_t root,
                                 id<MTLBuffer> level, id<MTLBuffer> frontierA, id<MTLBuffer> frontierB,
                                 id<MTLBuffer> numCurrFrontier, uint32_t* frontierSizes, std::string* error) {
    id<MTLBuffer> prevFrontier = frontierA;
    id<MTLBuffer> currFrontier = frontierB;
    ((uint32_t*)prevFrontier.contents)[0] = root;
    uint32_t numPrevFrontier = 1;
    if (frontierSizes) frontierSizes[0] = 1;
    // BLOCK_DIM threads per threadgroup, or fewer if this pipeline cannot run that many. The
    // kernel strides its commit loop by the threadgroup's real size, so fewer is safe.
    NSUInteger threads = std::min<NSUInteger>(BLOCK_DIM, pipeline.maxTotalThreadsPerThreadgroup);
    unsigned int launches = 0;
    for (uint32_t currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        *(uint32_t*)numCurrFrontier.contents = 0;
        @autoreleasepool {
            id<MTLCommandBuffer> cb = [queue commandBuffer];
            id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
            [enc setComputePipelineState:pipeline];
            [enc setBuffer:srcPtrs offset:0 atIndex:0];
            [enc setBuffer:dst offset:0 atIndex:1];
            [enc setBuffer:level offset:0 atIndex:2];
            [enc setBuffer:prevFrontier offset:0 atIndex:3];
            [enc setBuffer:currFrontier offset:0 atIndex:4];
            [enc setBytes:&numPrevFrontier length:sizeof(uint32_t) atIndex:5];
            [enc setBuffer:numCurrFrontier offset:0 atIndex:6];
            [enc setBytes:&currLevel length:sizeof(uint32_t) atIndex:7];
            // One thread per vertex of the previous frontier, in whole threadgroups.
            [enc dispatchThreadgroups:MTLSizeMake((numPrevFrontier + threads - 1) / threads, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
            [enc endEncoding];
            [cb commit];
            [cb waitUntilCompleted];
            ++launches;
            if (cb.status == MTLCommandBufferStatusError) {
                if (error && error->empty())
                    *error = std::string("command buffer error: ") +
                             (cb.error ? cb.error.localizedDescription.UTF8String : "unknown");
                return launches;
            }
        }
        uint32_t numCurr = *(const uint32_t*)numCurrFrontier.contents;
        if (frontierSizes) frontierSizes[currLevel] = numCurr;
        // A correct kernel finds each vertex once: stop instead of reading past the buffers.
        if (numCurr > numVertices || currLevel >= numVertices) break;
        numPrevFrontier = numCurr;
        std::swap(prevFrontier, currFrontier);
    }
    return launches;
}
