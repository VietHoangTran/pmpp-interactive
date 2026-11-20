// Lab 8.1 reference solution (Metal Shading Language).
// The same two kernels as kernel.cu, with the same differences from the page.
// The harness defines TILE_WIDTH when it compiles this file (it tests 8, 16 and,
// where the pipeline allows 1024 threads, 32).
//
// CUDA                         Metal
// __shared__ float a[..]       threadgroup float a[..]   (declared in the kernel body)
// __syncthreads()              threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx                    [[thread_position_in_threadgroup]]
// blockIdx                     [[threadgroup_position_in_grid]]
// blockIdx * blockDim + threadIdx   [[thread_position_in_grid]]
//
// Every position attribute is a uint3: MSL requires all of a kernel's position
// attributes to have the same number of components (notes/today/ch08-facts.md C2).
#include <metal_stdlib>
using namespace metal;

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// Same layout as ../stencil.h (a file compiled at run time cannot include it).
struct StencilCoeffs { float c0, c1, c2, c3, c4, c5, c6; };

// Fig. 8.6: one thread per grid point.
kernel void stencil_basic(device const float*    in [[buffer(0)]],
                          device float*         out [[buffer(1)]],
                          constant uint&          N [[buffer(2)]],
                          constant StencilCoeffs& c [[buffer(3)]],
                          uint3 gid [[thread_position_in_grid]]) {
    uint i = gid.z, j = gid.y, k = gid.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c.c0 * in[i * N * N + j * N + k]
                                   + c.c1 * in[i * N * N + j * N + (k - 1)]
                                   + c.c2 * in[i * N * N + j * N + (k + 1)]
                                   + c.c3 * in[i * N * N + (j - 1) * N + k]
                                   + c.c4 * in[i * N * N + (j + 1) * N + k]
                                   + c.c5 * in[(i - 1) * N * N + j * N + k]
                                   + c.c6 * in[(i + 1) * N * N + j * N + k];
    }
}

// Fig. 8.12: a 2D threadgroup marches through `depth` planes in z, with the previous,
// current and next values in registers and the current plane mirrored in
// threadgroup memory.
kernel void stencil_regtile(device const float*    in [[buffer(0)]],
                            device float*         out [[buffer(1)]],
                            constant uint&          N [[buffer(2)]],
                            constant uint&      depth [[buffer(3)]],
                            constant StencilCoeffs& c [[buffer(4)]],
                            uint3 tid [[thread_position_in_threadgroup]],
                            uint3 bid [[threadgroup_position_in_grid]]) {
    int iStart = int(bid.z * depth);
    int j = int(bid.y * OUT_TILE_DIM + tid.y) - 1;
    int k = int(bid.x * OUT_TILE_DIM + tid.x) - 1;
    int n = int(N);
    float inPrev = 0.0f;
    threadgroup float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr = 0.0f;
    float inNext = 0.0f;
    if (iStart - 1 >= 0 && iStart - 1 < n && j >= 0 && j < n && k >= 0 && k < n) {
        inPrev = in[(iStart - 1) * n * n + j * n + k];
    }
    if (iStart >= 0 && iStart < n && j >= 0 && j < n && k >= 0 && k < n) {
        inCurr = in[iStart * n * n + j * n + k];
        inCurr_s[tid.y][tid.x] = inCurr;
    }
    // The trip count is the same for every thread of the threadgroup, so both
    // barriers are reached by all of them on every iteration.
    for (int i = iStart; i < iStart + int(depth); ++i) {
        if (i + 1 >= 0 && i + 1 < n && j >= 0 && j < n && k >= 0 && k < n) {
            inNext = in[(i + 1) * n * n + j * n + k];
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (i >= 1 && i < n - 1 && j >= 1 && j < n - 1 && k >= 1 && k < n - 1) {
            if (tid.y >= 1 && tid.y < IN_TILE_DIM - 1 && tid.x >= 1 && tid.x < IN_TILE_DIM - 1) {
                out[i * n * n + j * n + k] = c.c0 * inCurr
                                           + c.c1 * inCurr_s[tid.y][tid.x - 1]
                                           + c.c2 * inCurr_s[tid.y][tid.x + 1]
                                           + c.c3 * inCurr_s[tid.y - 1][tid.x]
                                           + c.c4 * inCurr_s[tid.y + 1][tid.x]
                                           + c.c5 * inPrev
                                           + c.c6 * inNext;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[tid.y][tid.x] = inNext;
    }
}
