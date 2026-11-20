// Lab 15.6 starter (Metal Shading Language). Milestone M3:
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
// START FROM YOUR LAB 15.5 kernel.metal, with the changes kernel.cu lists: the X tile
// where A was, a new loadTileT that stores the W tile transposed into B_s[k][n], and
// LDB_S, B_s's own pitch. launch.mm is complete: it binds Y, X and W a second time as
// float4 (indices 4, 5, 6), in lab 15.5's order.
#include <metal_stdlib>
using namespace metal;

#ifndef SIMD_WIDTH
#error "the host defines SIMD_WIDTH, the SIMD-group width to lay the threadgroup out for"
#endif

#define bM 128          // block-level output tile: bM x bN (rows of Y = batch, columns = out)
#define bN 128
#define bK 8            // depth of the input tiles in threadgroup memory (along `in`)
#define LDA_S (bK + 1)  // leading dimension of A_s (the X tile): one padding column
#define LDB_S bN        // TODO: the pitch of B_s, as in kernel.cu
#define tM 8            // thread-level output tile (logical): tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

// Section 15.5's layout for a SIMD-group of SIMD_WIDTH threads.
#define LC (SIMD_WIDTH / 8)          // lanes per lane-row
#define QR ((SIMD_WIDTH / LC) * 4)   // quadrant height: 8 lane-rows of 4 rows
#define QC (LC * 4)                  // quadrant width
#define WC (bN / (2 * QC))           // SIMD-groups across the block tile

struct LinearDims { int batch; int in; int out; };   // must match launch.mm

METAL_FUNC void clear(thread float C_r[][tN], int m, int n) {
    // TODO: your lab 15.5 clear.
}

// Loads the X tile, as lab 15.5's loadTile loaded A. Element (row, col) of the tile is
// T[base + row * lda + col]; T4 is the same buffer as float4.
METAL_FUNC void loadTile(device const float* T, device const float4* T4, int base, int lda,
                         int maxRow, int maxCol, threadgroup float* T_s, int ldas,
                         int height, int width, int t) {
    // TODO: your lab 15.5 loadTile.
}

// New: loads the W tile, read as W is stored, element (row, col) to T_s[col * ldas + row].
METAL_FUNC void loadTileT(device const float* T, device const float4* T4, int base, int lda,
                          int maxRow, int maxCol, threadgroup float* T_s, int ldas,
                          int height, int width, int t) {
    // TODO
}

METAL_FUNC void mm(int k, threadgroup const float* a, int lda, threadgroup const float* b, int ldb,
                   thread float c[][tN]) {
    // TODO: your lab 15.5 mm.
}

METAL_FUNC void writeTile(device float* C, device float4* C4, int c0, int ldc, int maxRow, int maxCol,
                          thread float C_r[][tN]) {
    // TODO: your lab 15.5 writeTile.
}

[[max_total_threads_per_threadgroup(NUM_THREADS)]]
kernel void linear_m3(device const float*  X  [[buffer(0)]],
                      device const float*  W  [[buffer(1)]],
                      device float*        Y  [[buffer(2)]],
                      constant LinearDims& d  [[buffer(3)]],
                      device float4*       Y4 [[buffer(4)]],   // Y again, as float4
                      device const float4* X4 [[buffer(5)]],   // X again, as float4
                      device const float4* W4 [[buffer(6)]],   // W again, as float4
                      uint  tid [[thread_index_in_threadgroup]],
                      uint2 bid [[threadgroup_position_in_grid]]) {
    // TODO: your lab 15.5 kernel body, adapted as in kernel.cu. The threadgroup is
    // one-dimensional, 256 threads; bid.x walks the outputs, bid.y the batch.
}
