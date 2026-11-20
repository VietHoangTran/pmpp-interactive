// Lab 16.1 starter (Metal Shading Language).
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
// no built-in thread index either, so the helpers that need one take it as `tx`.
// The position attributes are unsigned: convert them to int before you subtract.
#include <metal_stdlib>
using namespace metal;

#if !defined(SW_MATCH) || !defined(SW_MISMATCH) || !defined(SW_GAP)
#error "the driver defines SW_MATCH, SW_MISMATCH and SW_GAP from sw.h"
#endif

// L = L_seq + 1, d = the front, firstRow = tile row of threadgroup 0, stride = W + pad.
struct SwParams { int L; int d; int firstRow; int stride; };

static inline int max4(int a, int b, int c, int d) {
    // TODO
    return 0;
}

// ---------------------------------------------------------------- square tiles
static inline int sqLoadN(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    // TODO
    return 0;
}
static inline int sqLoadW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    // TODO
    return 0;
}
static inline int sqLoadNW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                           int r_tile, int q_tile, int stride) {
    // TODO
    return 0;
}
static inline void sqStoreTile(device int* H, threadgroup const int* swTile, int L, int W, int stride,
                               int tile_row, int tile_col, int tx) {
    // TODO
}

kernel void sw_square(device int*        H      [[buffer(0)]],
                      device const int*  a      [[buffer(1)]],
                      device const int*  b      [[buffer(2)]],
                      constant SwParams& p      [[buffer(3)]],
                      threadgroup int*   swTile [[threadgroup(0)]],
                      uint tid [[thread_position_in_threadgroup]],
                      uint bid [[threadgroup_position_in_grid]],
                      uint tpg [[threads_per_threadgroup]]) {
    // TODO: TODO 1 to 3 of kernel.cu, in Metal's vocabulary.
}

// ---------------------------------------------------------------- sheared tiles
static inline void hyInitTile(threadgroup int* swTile, int W, int stride, int tx) {
    // TODO
}
static inline int hyLoadN(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    // TODO
    return 0;
}
static inline int hyLoadW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                          int r_tile, int q_tile, int stride) {
    // TODO
    return 0;
}
static inline int hyLoadNW(device const int* H, int r, int q, int L, threadgroup const int* swTile,
                           int r_tile, int q_tile, int stride) {
    // TODO
    return 0;
}
static inline void hyStoreTile(device int* H, threadgroup const int* swTile, int L, int W, int stride,
                               int tile_row, int tile_col, int tx) {
    // TODO
}

kernel void sw_hyper(device int*        H      [[buffer(0)]],
                     device const int*  a      [[buffer(1)]],
                     device const int*  b      [[buffer(2)]],
                     constant SwParams& p      [[buffer(3)]],
                     threadgroup int*   swTile [[threadgroup(0)]],
                     uint tid [[thread_position_in_threadgroup]],
                     uint bid [[threadgroup_position_in_grid]],
                     uint tpg [[threads_per_threadgroup]]) {
    // TODO: TODO 5 to 8 of kernel.cu, in Metal's vocabulary.
}
