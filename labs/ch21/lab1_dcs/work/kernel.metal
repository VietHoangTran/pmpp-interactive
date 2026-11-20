// Lab 21.1 starter (Metal Shading Language).
// The same three kernels as kernel.cu, in Metal's vocabulary. The signatures and buffer
// indices are given; they match launch.mm.
//
// CUDA                                    Metal
// __constant__ float atoms[...]           constant float4* atoms [[buffer(1)]], a buffer the
//   + cudaMemcpyToSymbol per chunk          host binds at a new offset for each chunk
// atoms[n], atoms[n+1], ...[n+3]           atoms[n].x, .y, .z, .w (one float4 per atom)
// atomicAdd(&grid[p], v)                  atomic_fetch_add_explicit(&grid[p], v, memory_order_relaxed)
//                                           on a grid declared device atomic_float*
// sqrtf                                   sqrt
// blockIdx, threadIdx                     [[threadgroup_position_in_grid]], [[thread_position_in_threadgroup]]
// blockDim.x                              [[dispatch_threads_per_threadgroup]].x
//
// Metal atomic functions take atomic types only, so the scatter kernel declares the grid
// `device atomic_float*`; the other two declare the same buffer `device float*`.
//
// The driver passes COARSEN (from dcs.h) when it compiles this file.
#include <metal_stdlib>
using namespace metal;

#ifndef COARSEN
#define COARSEN 4
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
    // TODO: kernel.cu's TODOs 1 to 3.
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
    // TODO: kernel.cu's TODOs 4 to 6. Metal hands you the global position directly.
}

// ------------------------------------------------- Fig. 21.10: coarsened, coalesced writes
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
    // TODO: kernel.cu's TODOs 7 to 9. The stride between a thread's points is the
    //       threadgroup's x size as dispatched, bdim.x.
}
