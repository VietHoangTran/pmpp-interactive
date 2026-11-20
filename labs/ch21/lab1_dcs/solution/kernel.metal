// Lab 21.1 reference solution (Metal Shading Language).
// The same three kernels as kernel.cu: Fig. 21.5 (scatter), Fig. 21.6 (gather) and
// Fig. 21.10 (coarsened, coalescing-aware), with the same corrections (see kernel.cu).
//
// CUDA                                    Metal
// __constant__ float atoms[...]           constant float4* atoms [[buffer(1)]], a buffer the
//   + cudaMemcpyToSymbol per chunk          host binds at a new offset for each chunk
// atoms[n], atoms[n+1], ...[n+3]           atoms[n].x, .y, .z, .w (one float4 per atom)
// atomicAdd(&grid[p], v)                  atomic_fetch_add_explicit(&grid[p], v, memory_order_relaxed)
//                                           on a grid declared device atomic_float*
// sqrtf                                   sqrt
// blockDim.x                              [[dispatch_threads_per_threadgroup]].x
//
// Metal atomic functions take atomic types only, so the scatter kernel declares the grid
// `device atomic_float*`; the other two declare the same buffer `device float*`. Float
// atomics need Metal 3 and an Apple7 (M1) or later GPU (infra-facts C1, ch21-facts C2).
// Apple documents no constant cache and no size limit for constant buffers (ch21-facts C1).
// MSL is not IEEE-strict by default, so results differ from a CPU reference in the last
// bits (ch05-facts C7, ch21-facts C3); the driver compares with a tolerance.
//
// The driver passes COARSEN (from dcs.h) when it compiles this file.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSEN
#define COARSEN 4
#endif
#if COARSEN != 4
#error "dcs_coarsened is written for four points per thread, as Fig. 21.10 is"
#endif

// ---------------------------------------------------------------- Fig. 21.5: scatter
kernel void dcs_scatter(device atomic_float* energygrid  [[buffer(0)]],
                        constant float4*     atoms       [[buffer(1)]],
                        constant int&        nx          [[buffer(2)]],
                        constant int&        ny          [[buffer(3)]],
                        constant float&      gridspacing [[buffer(4)]],
                        constant int&        k           [[buffer(5)]],
                        constant int&        numatoms    [[buffer(6)]],
                        uint                 n           [[thread_position_in_grid]]) {
    if (int(n) >= numatoms) return;   // past the end of the chunk
    float4 atom = atoms[n];           // x, y, z, charge
    float z = gridspacing * float(k);
    float dz = z - atom.z;
    float dz2 = dz * dz;
    int grid_slice_offset = nx * ny * k;
    float charge = atom.w;
    for (int j = 0; j < ny; j++) {
        float y = gridspacing * float(j);
        float dy = y - atom.y;
        float dy2 = dy * dy;
        int grid_row_offset = grid_slice_offset + nx * j;
        for (int i = 0; i < nx; i++) {
            float x = gridspacing * float(i);
            float dx = x - atom.x;
            atomic_fetch_add_explicit(&energygrid[grid_row_offset + i], charge / sqrt(dx * dx + dy2 + dz2),
                                      memory_order_relaxed);
        }
    }
}

// ---------------------------------------------------------------- Fig. 21.6: gather
kernel void dcs_gather(device float*    energygrid  [[buffer(0)]],
                       constant float4* atoms       [[buffer(1)]],
                       constant int&    nx          [[buffer(2)]],
                       constant int&    ny          [[buffer(3)]],
                       constant float&  gridspacing [[buffer(4)]],
                       constant int&    k           [[buffer(5)]],
                       constant int&    numatoms    [[buffer(6)]],
                       uint2            gid         [[thread_position_in_grid]]) {
    int i = int(gid.x);
    int j = int(gid.y);
    // launch.mm dispatches exactly nx x ny threads, so this never fires there; it keeps the
    // kernel correct under a dispatch rounded up to whole threadgroups.
    if (i >= nx || j >= ny) return;
    float z = gridspacing * float(k);
    float y = gridspacing * float(j);
    float x = gridspacing * float(i);
    float energy = 0.0f;
    for (int n = 0; n < numatoms; n++) {
        float4 atom = atoms[n];
        float dx = x - atom.x;
        float dy = y - atom.y;
        float dz = z - atom.z;
        energy += atom.w / sqrt(dx * dx + dy * dy + dz * dz);
    }
    energygrid[nx * ny * k + nx * j + i] += energy;
}

// ------------------------------------------------- Fig. 21.10: coarsened, coalesced writes
// The stride between a thread's points is the threadgroup's x size as dispatched. launch.mm
// dispatches whole threadgroups, so every threadgroup has that size (ch05-facts C4, C5).
kernel void dcs_coarsened(device float*    energygrid  [[buffer(0)]],
                          constant float4* atoms       [[buffer(1)]],
                          constant int&    nx          [[buffer(2)]],
                          constant int&    ny          [[buffer(3)]],
                          constant float&  gridspacing [[buffer(4)]],
                          constant int&    k           [[buffer(5)]],
                          constant int&    numatoms    [[buffer(6)]],
                          uint2 tid  [[thread_position_in_threadgroup]],
                          uint2 bid  [[threadgroup_position_in_grid]],
                          uint2 bdim [[dispatch_threads_per_threadgroup]]) {
    int i = int(bid.x * bdim.x * COARSEN + tid.x);
    int j = int(bid.y * bdim.y + tid.y);
    if (j >= ny) return;
    int stride = int(bdim.x);
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
        float dx1 = dx0 + stride * gridspacing;
        float dx2 = dx0 + 2 * stride * gridspacing;
        float dx3 = dx0 + 3 * stride * gridspacing;
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
    if (i + stride < nx) energygrid[row + i + stride] += energy1;
    if (i + 2 * stride < nx) energygrid[row + i + 2 * stride] += energy2;
    if (i + 3 * stride < nx) energygrid[row + i + 3 * stride] += energy3;
}
