// Lab 21.1, provided: dispatches the Metal baseline (coarse_adjacent.metal). Do not edit.
// The same chunk loop and the same whole-threadgroup grid as encodeDcsCoarsened in the
// reference solution, so the two kernels differ only in which points a thread owns.
#import <Metal/Metal.h>
#include <algorithm>
#include <cstdio>

#include "../lab1_dcs/dcs.h"

void encodeDcsCoarseAdjacent(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                             id<MTLBuffer> energygrid, int nx, int ny, int nz, float gridspacing, int k,
                             id<MTLBuffer> atoms, int numatoms, int chunk) {
    if (nx <= 0 || ny <= 0 || numatoms <= 0) return;
    if (k < 0 || k >= nz || chunk < 1 || chunk > CHUNK_MAX) {
        std::fprintf(stderr, "encodeDcsCoarseAdjacent: need 0 <= k < nz and 1 <= chunk <= %d\n", CHUNK_MAX);
        return;
    }
    [enc setComputePipelineState:pipeline];
    [enc setBuffer:energygrid offset:0 atIndex:0];
    [enc setBytes:&nx length:sizeof(nx) atIndex:2];
    [enc setBytes:&ny length:sizeof(ny) atIndex:3];
    [enc setBytes:&gridspacing length:sizeof(gridspacing) atIndex:4];
    [enc setBytes:&k length:sizeof(k) atIndex:5];
    const NSUInteger span = (NSUInteger)COARSEN * BLOCK_X;
    MTLSize groups = MTLSizeMake((nx + span - 1) / span, (ny + BLOCK_Y - 1) / BLOCK_Y, 1);
    for (int start = 0; start < numatoms; start += chunk) {
        int count = std::min(chunk, numatoms - start);
        [enc setBuffer:atoms offset:(NSUInteger)start * 16 atIndex:1];   // 16 bytes per atom
        [enc setBytes:&count length:sizeof(count) atIndex:6];
        [enc dispatchThreadgroups:groups threadsPerThreadgroup:MTLSizeMake(BLOCK_X, BLOCK_Y, 1)];
    }
}
