// Lab 15.4 reference solution (Metal Shading Language).
// Lab 15.3's kernel with one padding column in A_s and loadTile fetching four floats per
// device-memory load. Book: section 15.6 (pp. 361-363), Fig. 15.11; Exercise 1 (p. 370).
// The reasoning is in kernel.cu.
//
// Metal differences from kernel.cu (notes/today/ch15-facts.md):
// - Apple documents no threadgroup-memory banks (C2), so nothing documented predicts what
//   the padding does here: it is a measurement. MLX pads its threadgroup tiles by 16 bytes
//   in practice (C2, D3), which is practice, not documentation.
// - No pointer cast for the vector loads: the host binds A and B a second time as float4
//   (A4, B4), as it binds C for the stores (lab 6.2; ch06-facts.md C3). A tile element is
//   addressed by its element index in the matrix; its float4 is index / 4, which is exact
//   when the leading dimension is a multiple of 4.
// - The rest as in lab 15.3 (SIMD_WIDTH from the host, METAL_FUNC, the unroll pragma,
//   [[max_total_threads_per_threadgroup]], int indices; C1, C3, C4).
#include <metal_stdlib>
using namespace metal;

#ifndef SIMD_WIDTH
#error "the host defines SIMD_WIDTH, the SIMD-group width to lay the threadgroup out for"
#endif

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in threadgroup memory
#define LDA_S (bK + 1)  // leading dimension of A_s: bK plus one padding column
#define tM 8            // thread-level output tile (logical): tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

// Section 15.5's layout for a SIMD-group of SIMD_WIDTH threads (see kernel.cu).
#define LC (SIMD_WIDTH / 8)          // lanes per lane-row
#define QR ((SIMD_WIDTH / LC) * 4)   // quadrant height: 8 lane-rows of 4 rows
#define QC (LC * 4)                  // quadrant width
#define WC (bN / (2 * QC))           // SIMD-groups across the block tile

struct GemmSizes { int M; int K; int N; };

METAL_FUNC void clear(thread float C_r[][tN], int m, int n) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < m; ++row) {
        #pragma clang loop unroll(full)
        for (int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// Fig. 15.5 with vector loads (Exercise 1). The tile's element (row, col) is T[base +
// row * lda + col]; T4 is the same buffer as float4. One float4 per thread per tile when
// lda is a multiple of 4 and the four columns are inside the matrix; four scalar loads
// with zero fill otherwise. Scalar stores into threadgroup memory either way.
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

// Rank-1 updates over the four sub-tiles: per i, two 4-value pieces of A and two of B.
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

// Store the four sub-tiles. c0 is the element index in C of the top-left sub-tile's first
// element; maxRow and maxCol count the rows and columns of C from there. A sub-tile row
// goes out as one float4 through C4 when ldc is a multiple of 4 (then c0 + row * ldc + col
// is a multiple of 4) and all four columns are inside C; as scalars otherwise.
METAL_FUNC void writeTile(device float* C, device float4* C4, int c0, int ldc, int maxRow, int maxCol,
                          thread float C_r[][tN]) {
    const bool vec = (ldc % 4 == 0);
    #pragma clang loop unroll(full)
    for (int qr = 0; qr < 2; ++qr) {
        #pragma clang loop unroll(full)
        for (int r = 0; r < 4; ++r) {
            const int row = qr * QR + r;
            if (row >= maxRow) continue;
            #pragma clang loop unroll(full)
            for (int qc = 0; qc < 2; ++qc) {
                const int col = qc * QC;
                const int cr = qr * 4 + r, cc = qc * 4;
                const int e = c0 + row * ldc + col;
                if (vec && col + 3 < maxCol) {
                    C4[e / 4] = float4(C_r[cr][cc], C_r[cr][cc + 1], C_r[cr][cc + 2], C_r[cr][cc + 3]);
                } else {
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
kernel void gemm_padded(device const float*  A  [[buffer(0)]],
                        device const float*  B  [[buffer(1)]],
                        device float*        C  [[buffer(2)]],
                        constant GemmSizes&  s  [[buffer(3)]],
                        device float4*       C4 [[buffer(4)]],   // C again, as float4
                        device const float4* A4 [[buffer(5)]],   // A again, as float4
                        device const float4* B4 [[buffer(6)]],   // B again, as float4
                        uint  tid [[thread_index_in_threadgroup]],
                        uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float A_s[bM * LDA_S];   // the book: A_s[bM*(bK + 1)]
    threadgroup float B_s[bK * bN];

    const int M = s.M, K = s.K, N = s.N;
    const int t = int(tid);
    const int bRow = int(bid.y) * bM;
    const int bCol = int(bid.x) * bN;

    const int warp = t / SIMD_WIDTH;
    const int lane = t % SIMD_WIDTH;
    const int lRow = (lane / LC) * 4;
    const int lCol = (lane % LC) * 4;
    const int wRow = (warp / WC) * (2 * QR);
    const int wCol = (warp % WC) * (2 * QC);
    const int tRow = wRow + lRow;
    const int tCol = wCol + lCol;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    for (int tile = 0; tile < (K + bK - 1) / bK; ++tile) {
        loadTile(A, A4, bRow * K + tile * bK, K, M - bRow, K - tile * bK, A_s, LDA_S, bM, bK, t);
        loadTile(B, B4, tile * bK * N + bCol, N, K - tile * bK, N - bCol, B_s, bN, bK, bN, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm(bK, A_s + tRow * LDA_S, LDA_S, B_s + tCol, bN, C_r);   // the start offset too
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(C, C4, (bRow + tRow) * N + bCol + tCol, N, maxRow, maxCol, C_r);
}
