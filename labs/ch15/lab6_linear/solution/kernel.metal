// Lab 15.6 reference solution (Metal Shading Language). Milestone M3:
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
// Lab 15.5's kernel with the changes kernel.cu describes: loadTileT stores the W tile
// transposed into B_s, whose pitch is LDB_S = bN + 4 (ours). The reasoning is in
// kernel.cu.
//
// Metal differences from kernel.cu, as in lab 15.5 (notes/today/ch15-facts.md):
// - The host defines SIMD_WIDTH, the SIMD-group width to lay the threadgroup out for.
//   Apple documents no fixed width; the pipeline's threadExecutionWidth reports it (C1).
// - 16-byte accesses go through buffers declared `device float4*`, Metal's documented
//   route (ch06-facts C3), not a pointer cast: launch.mm binds Y, X and W a second time.
// - Indices are `int` (Apple advises signed indices, ch06-facts C3).
// - METAL_FUNC (always_inline) is Apple's header macro, not in the specification; the
//   Clang loop pragma is accepted by the compiler but not documented by Apple (C4).
// - Apple documents no threadgroup-memory banks (C2): both paddings follow the NVIDIA
//   analysis, and on Metal whether they help is a measurement.
#include <metal_stdlib>
using namespace metal;

#ifndef SIMD_WIDTH
#error "the host defines SIMD_WIDTH, the SIMD-group width to lay the threadgroup out for"
#endif

#define bM 128          // block-level output tile: bM x bN (rows of Y = batch, columns = out)
#define bN 128
#define bK 8            // depth of the input tiles in threadgroup memory (along `in`)
#define LDA_S (bK + 1)  // leading dimension of A_s (the X tile): one padding column
#define LDB_S (bN + 4)  // pitch of B_s (the W tile, stored transposed): ours
#define tM 8            // thread-level output tile (logical): tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

// Section 15.5's layout for a SIMD-group of SIMD_WIDTH threads (see kernel.cu).
#define LC (SIMD_WIDTH / 8)          // lanes per lane-row
#define QR ((SIMD_WIDTH / LC) * 4)   // quadrant height: 8 lane-rows of 4 rows
#define QC (LC * 4)                  // quadrant width
#define WC (bN / (2 * QC))           // SIMD-groups across the block tile

struct LinearDims { int batch; int in; int out; };   // must match launch.mm

METAL_FUNC void clear(thread float C_r[][tN], int m, int n) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < m; ++row) {
        #pragma clang loop unroll(full)
        for (int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// Lab 15.5's loadTile, unchanged (the X tile). Element (row, col) of the tile is
// T[base + row * lda + col]; T4 is the same buffer as float4.
METAL_FUNC void loadTile(device const float* T, device const float4* T4, int base, int lda,
                         int maxRow, int maxCol, threadgroup float* T_s, int ldas,
                         int height, int width, int t) {
    const int vecPerRow = width / 4;
    const int rowsPerSubTile = NUM_THREADS / vecPerRow;
    const int numSubTiles = height / rowsPerSubTile;
    const bool aligned = (lda % 4 == 0);   // then base and every row start are multiples of 4
    #pragma clang loop unroll(full)
    for (int subTile = 0; subTile < numSubTiles; ++subTile) {
        int row = subTile * rowsPerSubTile + t / vecPerRow;
        int col = (t % vecPerRow) * 4;
        int e = base + row * lda + col;
        threadgroup float* dst = T_s + (row * ldas + col);
        if (aligned && row < maxRow && col + 3 < maxCol) {
            float4 v = T4[e / 4];
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        } else {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                dst[j] = (row < maxRow && col + j < maxCol) ? T[e + j] : 0.0f;
            }
        }
    }
}

// The new helper: the W tile, read as W is stored and stored transposed, element
// (row, col) to T_s[col * ldas + row].
METAL_FUNC void loadTileT(device const float* T, device const float4* T4, int base, int lda,
                          int maxRow, int maxCol, threadgroup float* T_s, int ldas,
                          int height, int width, int t) {
    const int vecPerRow = width / 4;
    const int rowsPerSubTile = NUM_THREADS / vecPerRow;
    const int numSubTiles = height / rowsPerSubTile;
    const bool aligned = (lda % 4 == 0);
    #pragma clang loop unroll(full)
    for (int subTile = 0; subTile < numSubTiles; ++subTile) {
        int row = subTile * rowsPerSubTile + t / vecPerRow;
        int col = (t % vecPerRow) * 4;
        int e = base + row * lda + col;
        threadgroup float* dst = T_s + (col * ldas + row);
        if (aligned && row < maxRow && col + 3 < maxCol) {
            float4 v = T4[e / 4];
            dst[0 * ldas] = v.x;
            dst[1 * ldas] = v.y;
            dst[2 * ldas] = v.z;
            dst[3 * ldas] = v.w;
        } else {
            #pragma clang loop unroll(full)
            for (int j = 0; j < 4; ++j) {
                dst[j * ldas] = (row < maxRow && col + j < maxCol) ? T[e + j] : 0.0f;
            }
        }
    }
}

// Lab 15.5's mm, unchanged.
METAL_FUNC void mm(int k, threadgroup const float* a, int lda, threadgroup const float* b, int ldb,
                   thread float c[][tN]) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < k; ++i) {
        float a_r[tM];
        float b_r[tN];
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; ++r) {
            a_r[r] = a[r * lda + i];
            a_r[4 + r] = a[(QR + r) * lda + i];
        }
        #pragma clang loop unroll(full)
        for (int col = 0; col < 4; ++col) {
            b_r[col] = b[i * ldb + col];
            b_r[4 + col] = b[i * ldb + QC + col];
        }
        #pragma clang loop unroll(full)
        for (int row = 0; row < tM; ++row) {
            #pragma clang loop unroll(full)
            for (int col = 0; col < tN; ++col) {
                c[row][col] += a_r[row] * b_r[col];
            }
        }
    }
}

// Lab 15.5's writeTile. c0 is the element index in Y of the thread's top-left sub-tile;
// maxRow and maxCol count the rows and columns of Y from there.
METAL_FUNC void writeTile(device float* C, device float4* C4, int c0, int ldc, int maxRow, int maxCol,
                          thread float C_r[][tN]) {
    const bool vec = (ldc % 4 == 0);
    #pragma clang loop unroll(full)
    for (int qr = 0; qr < 2; ++qr) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; ++r) {
            const int row = qr * QR + r;
            #pragma clang loop unroll(full)
            for (int qc = 0; qc < 2; ++qc) {
                const int col = qc * QC;
                const int cr = qr * 4 + r, cc = qc * 4;
                const int e = c0 + row * ldc + col;
                if (row < maxRow && vec && col + 3 < maxCol) {
                    C4[e / 4] = float4(C_r[cr][cc], C_r[cr][cc + 1], C_r[cr][cc + 2], C_r[cr][cc + 3]);
                } else if (row < maxRow) {
                    #pragma clang loop unroll(full)
                    for (int j = 0; j < 4; ++j) {
                        if (col + j < maxCol) C[e + j] = C_r[cr][cc + j];
                    }
                }
            }
        }
    }
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
    // Two pairs of tiles: pair `tile` lives in buffers tile & 1.
    threadgroup float A_s[2][bM * LDA_S];   // X tile, padded
    threadgroup float B_s[2][bK * LDB_S];   // W tile, transposed: B_s[k][n]

    const int batch = d.batch, in = d.in, out = d.out;
    const int t = int(tid);
    const int bRow = int(bid.y) * bM;
    const int bCol = int(bid.x) * bN;

    const int warp = t / SIMD_WIDTH;
    const int lane = t % SIMD_WIDTH;
    const int tRow = (warp / WC) * (2 * QR) + (lane / LC) * 4;
    const int tCol = (warp % WC) * (2 * QC) + (lane % LC) * 4;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    const int numTiles = (in + bK - 1) / bK;
    if (numTiles > 0) {
        loadTile(X, X4, bRow * in, in, batch - bRow, in, A_s[0], LDA_S, bM, bK, t);
        loadTileT(W, W4, bCol * in, in, out - bCol, in, B_s[0], LDB_S, bN, bK, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int tile = 0; tile < numTiles - 1; ++tile) {
            const int cur = tile & 1, next = (tile + 1) & 1;
            mm(bK, A_s[cur] + tRow * LDA_S, LDA_S, B_s[cur] + tCol, LDB_S, C_r);
            // No barrier here: pair tile + 1 goes into the other buffers.
            const int k0 = (tile + 1) * bK;
            loadTile(X, X4, bRow * in + k0, in, batch - bRow, in - k0, A_s[next], LDA_S, bM, bK, t);
            loadTileT(W, W4, bCol * in + k0, in, out - bCol, in - k0, B_s[next], LDB_S, bN, bK, t);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const int last = (numTiles - 1) & 1;
        mm(bK, A_s[last] + tRow * LDA_S, LDA_S, B_s[last] + tCol, LDB_S, C_r);
    }

    const int maxRow = (bRow + tRow < batch) ? (batch - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < out) ? (out - (bCol + tCol)) : 0;
    writeTile(Y, Y4, (bRow + tRow) * out + bCol + tCol, out, maxRow, maxCol, C_r);
}
