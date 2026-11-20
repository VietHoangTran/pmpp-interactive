// Lab 15.2 reference solution (Metal Shading Language).
// Lab 15.1's kernel with the loops of mm interchanged (register tiling of the input
// tiles). Book: section 15.4 (pp. 357-360), Figs. 15.8 and 15.9.
//
// Only mm differs from lab 15.1; its Metal notes (METAL_FUNC, the unroll pragma,
// [[max_total_threads_per_threadgroup]], int indices) apply here unchanged
// (notes/today/ch15-facts.md C3, C4).
#include <metal_stdlib>
using namespace metal;

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in threadgroup memory
#define tM 8            // thread-level output tile: tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

struct GemmSizes { int M; int K; int N; };

// Fig. 15.4: zero the thread's tile, every index a constant.
METAL_FUNC void clear(thread float C_r[][tN], int m, int n) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < m; ++row) {
        #pragma clang loop unroll(full)
        for (int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// Fig. 15.5: the threadgroup copies a height x width tile into threadgroup memory, one
// sub-tile of NUM_THREADS elements at a time; elements outside the matrix become 0.
METAL_FUNC void loadTile(device const float* T, int lda, int maxRow, int maxCol,
                         threadgroup float* T_s, int ldas, int height, int width, int t) {
    const int rowsPerSubTile = NUM_THREADS / width;
    const int numSubTiles = height / rowsPerSubTile;
    #pragma clang loop unroll(full)
    for (int subTile = 0; subTile < numSubTiles; ++subTile) {
        int row = subTile * rowsPerSubTile + t / width;
        int col = t % width;
        if (row < maxRow && col < maxCol) {
            T_s[row * ldas + col] = T[row * lda + col];
        } else {
            T_s[row * ldas + col] = 0.0f;
        }
    }
}

// Fig. 15.9: k rank-1 updates from register strips; each threadgroup value is read
// once per tile pair.
METAL_FUNC void mm(int m, int n, int k, threadgroup const float* a, int lda,
                   threadgroup const float* b, int ldb, thread float c[][tN]) {
    #pragma clang loop unroll(full)
    for (int i = 0; i < k; ++i) {
        float a_r[tM];
        #pragma clang loop unroll(full)
        for (int row = 0; row < m; ++row) {
            a_r[row] = a[row * lda + i];
        }
        float b_r[tN];
        #pragma clang loop unroll(full)
        for (int col = 0; col < n; ++col) {
            b_r[col] = b[i * ldb + col];
        }
        #pragma clang loop unroll(full)
        for (int row = 0; row < m; ++row) {
            #pragma clang loop unroll(full)
            for (int col = 0; col < n; ++col) {
                c[row][col] += a_r[row] * b_r[col];
            }
        }
    }
}

// Fig. 15.7: store the thread's tile, skipping elements outside C.
METAL_FUNC void writeTile(device float* c, int ldc, int maxRow, int maxCol,
                          thread float C_r[][tN], int m, int n) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < m; ++row) {
        #pragma clang loop unroll(full)
        for (int col = 0; col < n; ++col) {
            if (row < maxRow && col < maxCol) {
                c[row * ldc + col] = C_r[row][col];
            }
        }
    }
}

[[max_total_threads_per_threadgroup(NUM_THREADS)]]
kernel void gemm_regtile(device const float* A [[buffer(0)]],
                         device const float* B [[buffer(1)]],
                         device float*       C [[buffer(2)]],
                         constant GemmSizes& s [[buffer(3)]],
                         uint  tid [[thread_index_in_threadgroup]],
                         uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float A_s[bM * bK];
    threadgroup float B_s[bK * bN];

    const int M = s.M, K = s.K, N = s.N;
    const int t = int(tid);
    const int bRow = int(bid.y) * bM;
    const int bCol = int(bid.x) * bN;
    const int tilesPerBlockX = bN / tN;
    const int ty = t / tilesPerBlockX;
    const int tx = t % tilesPerBlockX;
    const int tRow = ty * tM;
    const int tCol = tx * tN;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    for (int tile = 0; tile < (K + bK - 1) / bK; ++tile) {
        loadTile(A + (bRow * K + tile * bK), K, M - bRow, K - tile * bK, A_s, bK, bM, bK, t);
        loadTile(B + (tile * bK * N + bCol), N, K - tile * bK, N - bCol, B_s, bN, bK, bN, t);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm(tM, tN, bK, A_s + tRow * bK, bK, B_s + tCol, bN, C_r);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // A thread whose patch starts outside C gets maxRow or maxCol 0 and stores nothing.
    device float* c = C + ((bRow + tRow) * N + bCol + tCol);
    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(c, N, maxRow, maxCol, C_r, tM, tN);
}
