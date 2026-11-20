// Lab 15.3 starter (Metal Shading Language).
// Lab 15.2's solution, renamed: re-map the threads onto four 4 x 4 sub-tiles each and
// store each sub-tile row as one float4, as in kernel.cu (the layout is described there).
// Book: section 15.5 (pp. 360-361), Fig. 15.10; Exercises 3 and 2.
//
// Metal differences from kernel.cu (notes/today/ch15-facts.md):
// - The SIMD-group width is a property of the compiled pipeline, with no documented
//   value (C1). The host compiles this file with SIMD_WIDTH defined (the tests build 32
//   and 64; the benchmark uses the width the pipeline reports). Use SIMD_WIDTH where
//   kernel.cu uses the template parameter W.
// - No pointer cast for the float4 store. Apple's documented way to store four floats at
//   once is a buffer declared with a vector element type: bind C a second time as
//   `device float4*` (lab 6.2 did this) and index it in float4 units.
#include <metal_stdlib>
using namespace metal;

#ifndef SIMD_WIDTH
#error "the host defines SIMD_WIDTH, the SIMD-group width to lay the threadgroup out for"
#endif

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in threadgroup memory
#define tM 8            // thread-level output tile (logical): tM x tN
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

// TODO 2 (mm): lab 15.2's mm reads a contiguous 8 x 8 patch. Fill a_r with the thread's
//   4 + 4 rows (top and bottom quadrants) and b_r with its 4 + 4 columns (left and right
//   quadrants), as in kernel.cu.
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

// TODO 3 (writeTile): store the four 4 x 4 sub-tiles; a sub-tile row as one float4
//   through the float4 view of C when that is safe, as bounds-checked scalars otherwise.
//   Change the parameters as you need (the float4 view, an element index instead of a
//   pointer, ...).
METAL_FUNC void writeTile(device float* c, int ldc, int maxRow, int maxCol,
                          thread float C_r[][tN], int m, int n) {
}

[[max_total_threads_per_threadgroup(NUM_THREADS)]]
kernel void gemm_remap(device const float* A [[buffer(0)]],
                       device const float* B [[buffer(1)]],
                       device float*       C [[buffer(2)]],
                       constant GemmSizes& s [[buffer(3)]],
                       // TODO 3: C again, as float4, at a buffer index launch.mm binds
                       uint  tid [[thread_index_in_threadgroup]],
                       uint2 bid [[threadgroup_position_in_grid]]) {
    threadgroup float A_s[bM * bK];
    threadgroup float B_s[bK * bN];

    const int M = s.M, K = s.K, N = s.N;
    const int t = int(tid);
    const int bRow = int(bid.y) * bM;
    const int bCol = int(bid.x) * bN;
    // TODO 1 (mapping): replace lab 15.2's mapping with section 15.5's, for a SIMD-group
    //   of SIMD_WIDTH threads (as in kernel.cu).
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
        mm(tM, tN, bK, A_s + tRow * bK, bK, B_s + tCol, bN, C_r);   // TODO 2: the new mm
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // A thread whose patch starts outside C gets maxRow or maxCol 0 and stores nothing.
    device float* c = C + ((bRow + tRow) * N + bCol + tCol);
    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(c, N, maxRow, maxCol, C_r, tM, tN);                 // TODO 3: the new writeTile
}
