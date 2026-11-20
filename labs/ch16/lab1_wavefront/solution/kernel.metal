// Lab 16.1 reference solution (Metal Shading Language).
// The same two kernels as kernel.cu: square tiles (book section 16.6, Figs. 16.9 to
// 16.12) and sheared tiles (section 16.7, Figs. 16.16 to 16.19). The driver compiles
// this file with SW_MATCH, SW_MISMATCH and SW_GAP defined from ../sw.h.
//
// CUDA                               Metal
// extern __shared__ int swTile[]     threadgroup int* swTile [[threadgroup(0)]], sized by the host
// __syncthreads()                    threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x                        [[thread_position_in_threadgroup]]
// blockIdx.x                         [[threadgroup_position_in_grid]]
// blockDim.x                         [[threads_per_threadgroup]]
//
// A helper cannot declare threadgroup memory, so the kernel passes the tile in; it has
// no built-in thread index either, so the helpers that need one take it as `tx`
// (notes/today/ch16-facts.md C3). The barrier orders threadgroup memory only, which is
// enough: the neighbours read from device memory were written by earlier dispatches.
#include <metal_stdlib>
using namespace metal;

#if !defined(SW_MATCH) || !defined(SW_MISMATCH) || !defined(SW_GAP)
#error "the driver defines SW_MATCH, SW_MISMATCH and SW_GAP from sw.h"
#endif

// L = L_seq + 1, d = the front, firstRow = tile row of threadgroup 0, stride = W + pad.
struct SwParams { int L; int d; int firstRow; int stride; };

static inline int max4(int a, int b, int c, int d) {
    int m = a;
    if (m < b) m = b;
    if (m < c) m = c;
    if (m < d) m = d;
    return m;
}

// ---------------------------------------------------------------- square tiles
static inline int sqLoadN(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    return (r_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + q_tile];
}
static inline int sqLoadW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
static inline int sqLoadNW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                           int r_tile, int q_tile, int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + (q - 1)] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}

// The thread index is the tile column here.
static inline void sqStoreTile(device int* H, threadgroup const int* swTile, int L, int W, int stride,
                               int tile_row, int tile_col, int tx) {
    for (int row = 0; row < W; row++) {
        const int r = W * tile_row + row + 1;
        const int q = W * tile_col + tx + 1;
        if (r < L && q < L) H[r * L + q] = swTile[row * stride + tx];
    }
}

kernel void sw_square(device int*        H      [[buffer(0)]],
                      device const int*  a      [[buffer(1)]],
                      device const int*  b      [[buffer(2)]],
                      constant SwParams& p      [[buffer(3)]],
                      threadgroup int*   swTile [[threadgroup(0)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint bid [[threadgroup_position_in_grid]],
                      uint tpg [[threads_per_threadgroup]]) {
    const int L = p.L, stride = p.stride;
    const int W = int(tpg);
    const int numTiles = (L - 1 + W - 1) / W;
    const int tile_row = p.firstRow + int(bid);
    const int tile_col = p.d - tile_row;
    if (tile_col >= 0 && tile_col < numTiles) {
        const int r_tile = int(tid);
        const int r = W * tile_row + r_tile + 1;
        for (int d_tile = 0; d_tile < 2 * W - 1; d_tile++) {
            const int q_tile = d_tile - r_tile;
            const int q = W * tile_col + q_tile + 1;
            if (q_tile >= 0 && q_tile < W && r < L && q < L) {
                const int n = sqLoadN(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int w = sqLoadW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int nw = sqLoadNW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int s = (a[r - 1] == b[q - 1]) ? SW_MATCH : SW_MISMATCH;
                swTile[r_tile * stride + q_tile] = max4(0, nw + s, w + SW_GAP, n + SW_GAP);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);   // every thread, every step
        }
        sqStoreTile(H, swTile, L, W, stride, tile_row, tile_col, int(tid));
    }
}

// ---------------------------------------------------------------- sheared tiles
static inline void hyInitTile(threadgroup int* swTile, int W, int stride, int tx) {
    for (int row = 0; row < W; row++) swTile[row * stride + tx] = 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

static inline int hyLoadN(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}
static inline int hyLoadW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
static inline int hyLoadNW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                           int r_tile, int q_tile, int stride) {
    return (r_tile == 0 || q_tile == 0 || q_tile == 1) ? H[(r - 1) * L + (q - 1)]
                                                       : swTile[(r_tile - 1) * stride + (q_tile - 2)];
}

static inline void hyStoreTile(device int* H, threadgroup const int* swTile, int L, int W, int stride,
                               int tile_row, int tile_col, int tx) {
    for (int row = 0; row < W; row++) {
        const int r = W * tile_row + row + 1;
        const int q = W * tile_col + tx - row + 1;
        if (r < L && q >= 1 && q < L) H[r * L + q] = swTile[row * stride + tx];
    }
}

kernel void sw_hyper(device int*        H      [[buffer(0)]],
                     device const int*  a      [[buffer(1)]],
                     device const int*  b      [[buffer(2)]],
                     constant SwParams& p      [[buffer(3)]],
                     threadgroup int*   swTile [[threadgroup(0)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint bid [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    const int L = p.L, stride = p.stride;
    const int W = int(tpg);
    const int numTiles = (L - 1 + W - 1) / W;
    const int tile_row = p.firstRow + int(bid);
    const int tile_col = p.d - 2 * tile_row;
    if (tile_col >= 0 && tile_col <= numTiles) {
        hyInitTile(swTile, W, stride, int(tid));
        const int r_tile = int(tid);
        const int r = W * tile_row + r_tile + 1;
        for (int d_tile = 0; d_tile < W; d_tile++) {
            const int q_tile = d_tile;
            const int q = W * tile_col + q_tile - r_tile + 1;
            if (r < L && q >= 1 && q < L) {
                const int n = hyLoadN(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int w = hyLoadW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int nw = hyLoadNW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int s = (a[r - 1] == b[q - 1]) ? SW_MATCH : SW_MISMATCH;
                swTile[r_tile * stride + q_tile] = max4(0, nw + s, w + SW_GAP, n + SW_GAP);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        hyStoreTile(H, swTile, L, W, stride, tile_row, tile_col, int(tid));
    }
}
