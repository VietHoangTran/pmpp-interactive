// Lab 21.1, provided: the benchmark's baseline for your coarsened kernel. Do not edit.
// (Metal Shading Language.) The Metal counterpart of coarse_adjacent.cu: book Fig. 21.8
// (p. 520) with line 05 corrected, so each thread owns four ADJACENT points of a row.
// See coarse_adjacent.cu for what changed from the page and why.
//
// The driver passes COARSEN (from dcs.h) when it compiles this file.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSEN
#define COARSEN 4
#endif
#if COARSEN != 4
#error "dcs_coarse_adjacent is written for four points per thread, as Fig. 21.8 is"
#endif

kernel void dcs_coarse_adjacent(device float*    energygrid  [[buffer(0)]],
                                constant float4* atoms       [[buffer(1)]],
                                constant int&    nx          [[buffer(2)]],
                                constant int&    ny          [[buffer(3)]],
                                constant float&  gridspacing [[buffer(4)]],
                                constant int&    k           [[buffer(5)]],
                                constant int&    numatoms    [[buffer(6)]],
                                uint2 tid  [[thread_position_in_threadgroup]],
                                uint2 bid  [[threadgroup_position_in_grid]],
                                uint2 bdim [[dispatch_threads_per_threadgroup]]) {
    int i = int(bid.x * bdim.x * COARSEN + tid.x * COARSEN);   // line 05, corrected
    int j = int(bid.y * bdim.y + tid.y);
    if (j >= ny) return;
    float z = gridspacing * float(k);
    float y = gridspacing * float(j);
    float x = gridspacing * float(i);
    float energy0 = 0.0f;
    float energy1 = 0.0f;
    float energy2 = 0.0f;
    float energy3 = 0.0f;
    for (int n = 0; n < numatoms; n++) {
        float4 atom = atoms[n];
        float dx0 = x - atom.x;
        float dx1 = dx0 + gridspacing;
        float dx2 = dx0 + 2 * gridspacing;
        float dx3 = dx0 + 3 * gridspacing;
        float dy = y - atom.y;
        float dz = z - atom.z;
        float dysqdzsq = dy * dy + dz * dz;
        float charge = atom.w;
        energy0 += charge / sqrt(dx0 * dx0 + dysqdzsq);
        energy1 += charge / sqrt(dx1 * dx1 + dysqdzsq);
        energy2 += charge / sqrt(dx2 * dx2 + dysqdzsq);
        energy3 += charge / sqrt(dx3 * dx3 + dysqdzsq);
    }
    int row = nx * ny * k + nx * j;
    if (i < nx) energygrid[row + i] += energy0;
    if (i + 1 < nx) energygrid[row + i + 1] += energy1;
    if (i + 2 < nx) energygrid[row + i + 2] += energy2;
    if (i + 3 < nx) energygrid[row + i + 3] += energy3;
}
