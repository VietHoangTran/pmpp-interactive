// Lab 15.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Large output tiles through thread coarsening: a block of 256 threads computes a
// 128 x 128 tile of C, and each thread an 8 x 8 patch of it, held in registers.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.3 (pp. 352-357), Figs. 15.3 to 15.7. Milestone M3, stage 1 of 5.
//
// Where this file differs from the page:
// - The block is one-dimensional, 256 threads. The prose describes 16 x 16 threads
//   (p. 352), but Fig. 15.3 computes both of a thread's coordinates from threadIdx.x
//   alone, which only works for a 1D block (notes/ch15.md, "Unclear" 7). The 16 x 16
//   survives as the logical arrangement: ty = t / 16, tx = t % 16.
// - A_s and B_s are declared once, at the top of the kernel. Fig. 15.3 declares them
//   inside the tile loop (lines 23 and 24).
// - Sizes are int and the arguments are (A, B, C, M, K, N): lab 5.3's order (A's rows,
//   the inner dimension, B's columns), so milestone M2's call sites work unchanged.
//   Fig. 15.3 has unsigned int and the order (A, B, C, M, N, K). A and B are const.
// - The helpers are static as well as __forceinline__. Later stages link this solution
//   into their benchmark. In the CPU emulator's build two non-static inline functions
//   both named mm, with different bodies, would be merged by the linker without a word.
// - __launch_bounds__(NUM_THREADS) is ours, not the book's. It tells the compiler the
//   largest block the kernel is launched with, so it can budget registers for 256
//   threads. Without it HIP compiles for the largest block the device accepts
//   (notes/today/ch15-facts.md B3, ch08-facts.md A4 and B4).
// - The signatures of Figs. 15.6 and 15.7 are clipped in the PDF (notes, "Unclear" 1
//   and 2); these are complete.

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in shared memory
#define tM 8            // thread-level output tile: tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

// Fig. 15.4: zero the thread's tile. Full unrolling makes every index a constant, which
// is what lets the compiler keep C_r in registers (p. 353).
static __device__ __forceinline__ void clear(float C_r[][tN], int m, int n) {
    #pragma unroll
    for (int row = 0; row < m; ++row) {
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// Fig. 15.5: the block copies a height x width tile of a matrix (leading dimension lda)
// into shared memory (leading dimension ldas), one sub-tile of NUM_THREADS elements at a
// time. maxRow and maxCol count the rows and columns left in the matrix from the tile's
// corner; elements past them are written as 0, so mm needs no bounds checks.
static __device__ __forceinline__ void loadTile(const float* T, int lda, int maxRow, int maxCol,
                                                float* T_s, int ldas, int height, int width) {
    const int rowsPerSubTile = NUM_THREADS / width;
    const int numSubTiles = height / rowsPerSubTile;
    #pragma unroll
    for (int subTile = 0; subTile < numSubTiles; ++subTile) {
        int row = subTile * rowsPerSubTile + threadIdx.x / width;
        int col = threadIdx.x % width;
        if (row < maxRow && col < maxCol) {
            T_s[row * ldas + col] = T[row * lda + col];
        } else {
            T_s[row * ldas + col] = 0.0f;
        }
    }
}

// Fig. 15.6: add one tile pair's contribution to the thread's m x n tile. a points at the
// thread's first row of A_s, b at its first column of B_s. Every product reads both
// operands from shared memory.
static __device__ __forceinline__ void mm(int m, int n, int k, const float* a, int lda,
                                          const float* b, int ldb, float c[][tN]) {
    #pragma unroll
    for (int row = 0; row < m; ++row) {
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            #pragma unroll
            for (int i = 0; i < k; ++i) {
                c[row][col] += a[row * lda + i] * b[i * ldb + col];
            }
        }
    }
}

// Fig. 15.7: store the thread's tile, skipping elements outside C.
static __device__ __forceinline__ void writeTile(float* c, int ldc, int maxRow, int maxCol,
                                                 float C_r[][tN], int m, int n) {
    #pragma unroll
    for (int row = 0; row < m; ++row) {
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            if (row < maxRow && col < maxCol) {
                c[row * ldc + col] = C_r[row][col];
            }
        }
    }
}

__global__ void __launch_bounds__(NUM_THREADS)
gemmCoarseKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float A_s[bM * bK];
    __shared__ float B_s[bK * bN];

    // Where this block's tile starts in C (Fig. 15.3, lines 05-06).
    const int bRow = blockIdx.y * bM;
    const int bCol = blockIdx.x * bN;
    // Where this thread's patch starts inside the block's tile (lines 09-13).
    const int tilesPerBlockX = bN / tN;
    const int ty = threadIdx.x / tilesPerBlockX;
    const int tx = threadIdx.x % tilesPerBlockX;
    const int tRow = ty * tM;
    const int tCol = tx * tN;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    // One iteration per pair of input tiles: a bM x bK tile of A and a bK x bN tile of B.
    for (int tile = 0; tile < (K + bK - 1) / bK; ++tile) {
        loadTile(&A[bRow * K + tile * bK], K, M - bRow, K - tile * bK, &A_s[0], bK, bM, bK);
        loadTile(&B[tile * bK * N + bCol], N, K - tile * bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();   // the tiles are complete before anyone reads them
        mm(tM, tN, bK, &A_s[tRow * bK], bK, &B_s[tCol], bN, C_r);
        __syncthreads();   // everyone has finished reading before the next loads overwrite them
    }

    // A thread whose patch starts outside C gets maxRow or maxCol 0 and stores nothing.
    float* c = &C[(bRow + tRow) * N + bCol + tCol];
    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(c, N, maxRow, maxCol, C_r, tM, tN);
}

void launchGemmCoarse(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 grid((N + bN - 1) / bN,    // x covers the columns of C
              (M + bM - 1) / bM);   // y covers the rows of C
    gemmCoarseKernel<<<grid, NUM_THREADS>>>(A, B, C, M, K, N);
}
