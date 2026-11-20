// Lab 15.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Register tiling of the input tiles: lab 15.1's kernel with the loops of mm interchanged,
// so that each value a thread needs from shared memory is read once per tile pair and
// reused from a register.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.4 (pp. 357-360), Figs. 15.8 and 15.9. Milestone M3, stage 2 of 5.
//
// Only mm differs from lab 15.1. Where this file differs from the page (the reasons are
// in lab 15.1's solution):
// - a one-dimensional block of 256 threads (the prose says 16 x 16, the code needs 1D);
// - A_s and B_s declared once at the top of the kernel, not inside the tile loop;
// - int sizes, const inputs, and lab 5.3's argument order (A, B, C, M, K, N);
// - static helpers, and __launch_bounds__(NUM_THREADS), which is ours.

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

// Fig. 15.9: add one tile pair's contribution to the thread's m x n tile, as k rank-1
// updates. For each i the thread copies one column strip of its rows of A_s (m values)
// and one row strip of its columns of B_s (n values) into registers, then adds their
// outer product to c. Each shared value is read once per tile pair instead of n (for A)
// or m (for B) times. All five loops are unrolled, so a_r, b_r and c have constant
// indices and stay in registers (p. 359).
static __device__ __forceinline__ void mm(int m, int n, int k, const float* a, int lda,
                                          const float* b, int ldb, float c[][tN]) {
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        // Load the A strip to registers.
        float a_r[tM];
        #pragma unroll
        for (int row = 0; row < m; ++row) {
            a_r[row] = a[row * lda + i];
        }
        // Load the B strip to registers.
        float b_r[tN];
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            b_r[col] = b[i * ldb + col];
        }
        // Compute with the strips.
        #pragma unroll
        for (int row = 0; row < m; ++row) {
            #pragma unroll
            for (int col = 0; col < n; ++col) {
                c[row][col] += a_r[row] * b_r[col];
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
gemmRegTileKernel(const float* A, const float* B, float* C, int M, int K, int N) {
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

void launchGemmRegTile(const float* A, const float* B, float* C, int M, int K, int N) {
    dim3 grid((N + bN - 1) / bN,    // x covers the columns of C
              (M + bM - 1) / bM);   // y covers the rows of C
    gemmRegTileKernel<<<grid, NUM_THREADS>>>(A, B, C, M, K, N);
}
