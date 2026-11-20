// Lab 21.1 starter: the Metal counterparts of launchDcsScatter, launchDcsGather and
// launchDcsCoarsened. Each encodes its whole chunk loop into the open encoder it is given
// and commits nothing.
//
// There is no symbol to copy to on Metal. All the atoms sit in one buffer, 16 bytes per
// atom (a float4). A chunk is that same buffer bound again with
//   [enc setBuffer:atoms offset:<bytes to the chunk's first atom> atIndex:1];
// before the chunk's dispatch. The dispatches of one plain compute encoder run in the
// order they were encoded, and each sees the previous one's writes (ch02-facts C9).
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>

#include "../dcs.h"

// Buffer indices 0 and 2..5: the grid and the slice description, the same for every chunk.
static void bindSlice(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> energygrid, int nx, int ny,
                      float gridspacing, int k) {
    [enc setBuffer:energygrid offset:0 atIndex:0];
    [enc setBytes:&nx length:sizeof(nx) atIndex:2];
    [enc setBytes:&ny length:sizeof(ny) atIndex:3];
    [enc setBytes:&gridspacing length:sizeof(gridspacing) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
}

// ---------------------------------------------------------------- scatter
void encodeDcsScatter(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                      id<MTLBuffer> atoms, int numatoms, int chunk) {
    [enc setComputePipelineState:pipeline];
    bindSlice(enc, energygrid, nx, ny, gridspacing, k);
    // TODO: for each chunk of at most `chunk` atoms: bind its atoms at index 1, its atom
    //       count at index 6 (setBytes), and dispatch one thread per atom of the chunk in
    //       threadgroups of BLOCK_X. dispatchThreads makes exactly the number of threads
    //       you ask for (ch05-facts C5):
    //   [enc dispatchThreads:MTLSizeMake(?, 1, 1) threadsPerThreadgroup:MTLSizeMake(?, 1, 1)];
}

// ---------------------------------------------------------------- gather
void encodeDcsGather(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                     id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                     id<MTLBuffer> atoms, int numatoms, int chunk) {
    [enc setComputePipelineState:pipeline];
    bindSlice(enc, energygrid, nx, ny, gridspacing, k);
    // TODO: the same chunk loop, one thread per point, threadgroups of BLOCK_X x BLOCK_Y.
}

// ---------------------------------------------------------------- coarsened
void encodeDcsCoarsened(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                        id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                        id<MTLBuffer> atoms, int numatoms, int chunk) {
    [enc setComputePipelineState:pipeline];
    bindSlice(enc, energygrid, nx, ny, gridspacing, k);
    // TODO: the same chunk loop, threadgroups of BLOCK_X x BLOCK_Y, each covering
    //       COARSEN * BLOCK_X points of a row. Choose between dispatchThreads and
    //       dispatchThreadgroups:threadsPerThreadgroup: with care. dispatchThreads trims
    //       the threadgroups at the edge of the grid. What would a trimmed threadgroup
    //       leave uncomputed, given which points your kernel gives each thread?
}
