// Lab 15.1 reference solution (Metal Shading Language).
// The same kernel as kernel.cu: a threadgroup of 256 threads computes a 128 x 128 tile
// of C, each thread an 8 x 8 patch of it held in registers.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.3 (pp. 352-357), Figs. 15.3 to 15.7.
//
// Metal differences from kernel.cu (notes/today/ch15-facts.md):
// - The helpers are METAL_FUNC, Apple's macro for inline __attribute__((__always_inline__)),
//   defined in <metal_types>, which <metal_stdlib> includes. The MSL specification has no
//   inline attribute; the macro is in Apple's shipped headers, and MLX uses it (C4).
// - Loops are unrolled with `#pragma clang loop unroll(full)`. Apple documents no unroll
//   control; the compiler accepts this Clang pragma and MLX uses it. Whether the GPU
//   compiler honours it is not documented (C4).
// - [[max_total_threads_per_threadgroup(256)]] is the counterpart of __launch_bounds__.
//   Metal may give a pipeline a lower thread limit than asked for, so the driver checks
//   maxTotalThreadsPerThreadgroup >= 256 before using it (C3).
// - Helpers cannot read the thread's index from a built-in variable, so the kernel
//   passes it in (t).
// - Indices are int, not uint, as in lab 6.2: Apple says uint arithmetic must wrap, which
//   can stop the compiler from vectorising loads (ch06-facts C3).
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

// Fig. 15.6: one tile pair's contribution; every product reads threadgroup memory twice.
METAL_FUNC void mm(int m, int n, int k, threadgroup const float* a, int lda,
                   threadgroup const float* b, int ldb, thread float c[][tN]) {
    #pragma clang loop unroll(full)
    for (int row = 0; row < m; ++row) {
        #pragma clang loop unroll(full)
        for (int col = 0; col < n; ++col) {
            #pragma clang loop unroll(full)
            for (int i = 0; i < k; ++i) {
                c[row][col] += a[row * lda + i] * b[i * ldb + col];
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
kernel void gemm_coarse(device const float* A [[buffer(0)]],
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
