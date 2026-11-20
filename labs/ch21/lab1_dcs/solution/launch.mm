// Lab 21.1 reference solution: the Metal counterparts of launchDcsScatter, launchDcsGather
// and launchDcsCoarsened. Each encodes its whole chunk loop into the open encoder it is
// given and commits nothing.
//
// There is no symbol to copy to on Metal. All the atoms sit in one buffer, 16 bytes per
// atom (a float4), and each chunk is that buffer bound again at the chunk's first atom:
// offset 16 * start. That is what replaces cudaMemcpyToSymbol (ch21-facts C1). Apple
// requires a constant-buffer offset to be a multiple of 4 bytes (infra-facts C6), and a
// float4 needs 16; 16 * start is both.
//
// The dispatches of one plain compute encoder run in the order they were encoded, and the
// grid buffer is hazard-tracked, so each chunk's dispatch sees the previous chunk's
// additions (ch02-facts C9). Not in the book.
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>

#include "../dcs.h"

static bool argsOk(const char* who, int nx, int ny, int nz, int k, int numatoms, int chunk) {
    if (nx <= 0 || ny <= 0 || numatoms <= 0) return false;
    if (k < 0 || k >= nz || chunk < 1 || chunk > CHUNK_MAX) {
        std::fprintf(stderr, "%s: need 0 <= k < nz and 1 <= chunk <= %d (got k %d, nz %d, chunk %d)\n", who,
                     CHUNK_MAX, k, nz, chunk);
        return false;
    }
    return true;
}

// Buffer indices 0 and 2..5: the grid and the slice description, the same for every chunk.
static void bindSlice(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> energygrid, int nx, int ny,
                      float gridspacing, int k) {
    [enc setBuffer:energygrid offset:0 atIndex:0];
    [enc setBytes:&nx length:sizeof(nx) atIndex:2];
    [enc setBytes:&ny length:sizeof(ny) atIndex:3];
    [enc setBytes:&gridspacing length:sizeof(gridspacing) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
}

// Buffer indices 1 and 6: this chunk's atoms and how many there are.
static void bindChunk(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> atoms, int start, int count) {
    [enc setBuffer:atoms offset:(NSUInteger)start * 16 atIndex:1];
    [enc setBytes:&count length:sizeof(count) atIndex:6];
}

// ---------------------------------------------------------------- scatter
// One thread per atom of the chunk, in 1D threadgroups of BLOCK_X. dispatchThreads makes
// exactly `count` threads (Metal trims the last threadgroup, ch05-facts C5).
void encodeDcsScatter(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                      id<MTLBuffer> atoms, int numatoms, int chunk) {
    if (!argsOk("encodeDcsScatter", nx, ny, nz, k, numatoms, chunk)) return;
    [enc setComputePipelineState:pipeline];
    bindSlice(enc, energygrid, nx, ny, gridspacing, k);
    for (int start = 0; start < numatoms; start += chunk) {
        int count = std::min(chunk, numatoms - start);
        bindChunk(enc, atoms, start, count);
        [enc dispatchThreads:MTLSizeMake(count, 1, 1) threadsPerThreadgroup:MTLSizeMake(BLOCK_X, 1, 1)];
    }
}

// ---------------------------------------------------------------- gather
// One thread per grid point: exactly nx x ny threads in BLOCK_X x BLOCK_Y threadgroups.
void encodeDcsGather(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                     id<MTLBuffer> atoms, int numatoms, int chunk) {
    if (!argsOk("encodeDcsGather", nx, ny, nz, k, numatoms, chunk)) return;
    [enc setComputePipelineState:pipeline];
    bindSlice(enc, energygrid, nx, ny, gridspacing, k);
    for (int start = 0; start < numatoms; start += chunk) {
        int count = std::min(chunk, numatoms - start);
        bindChunk(enc, atoms, start, count);
        [enc dispatchThreads:MTLSizeMake(nx, ny, 1) threadsPerThreadgroup:MTLSizeMake(BLOCK_X, BLOCK_Y, 1)];
    }
}

// ---------------------------------------------------------------- coarsened
// Whole threadgroups, as in CUDA. dispatchThreads would trim the threadgroups at the end of
// a row, and a trimmed threadgroup's threads t < size cover only points base + t + m * BLOCK_X
// for those t: the points owned by the missing threads would never be computed.
void encodeDcsCoarsened(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                        id<MTLBuffer> atoms, int numatoms, int chunk) {
    if (!argsOk("encodeDcsCoarsened", nx, ny, nz, k, numatoms, chunk)) return;
    [enc setComputePipelineState:pipeline];
    bindSlice(enc, energygrid, nx, ny, gridspacing, k);
    const NSUInteger span = (NSUInteger)COARSEN * BLOCK_X;   // points of a row per threadgroup
    MTLSize groups = MTLSizeMake((nx + span - 1) / span, (ny + BLOCK_Y - 1) / BLOCK_Y, 1);
    for (int start = 0; start < numatoms; start += chunk) {
        int count = std::min(chunk, numatoms - start);
        bindChunk(enc, atoms, start, count);
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(BLOCK_X, BLOCK_Y, 1)];
    }
}
