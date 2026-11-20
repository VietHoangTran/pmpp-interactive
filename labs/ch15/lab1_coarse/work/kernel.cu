// Lab 15.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Large output tiles through thread coarsening: a block of 256 threads computes a
// 128 x 128 tile of C, and each thread an 8 x 8 patch of it, held in registers.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.3 (pp. 352-357), Figs. 15.3 to 15.7. Milestone M3, stage 1 of 5.
//
// Write the four helpers and the kernel the way the book does, then the launch.
// Where this lab differs from the page, on purpose:
// - The block is one-dimensional: 256 threads, numbered 0 to 255. The prose says
//   16 x 16, but Fig. 15.3 reads only threadIdx.x.
// - Declare A_s and B_s once, at the top of the kernel, not inside the tile loop.
// - Sizes are int, and the arguments are (A, B, C, M, K, N): lab 5.3's order.
// - Keep the helpers static: later stages link the solution of this lab into their
//   benchmark, and the CPU emulator's linker would silently merge two non-static inline
//   functions that share a name.
// - Keep __launch_bounds__(NUM_THREADS) on the kernel (ours, not the book's): it tells
//   the compiler the block size, so it can give each thread the registers it needs.

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in shared memory
#define tM 8            // thread-level output tile: tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

// Fig. 15.4. Set every element of the m x n tile to zero.
// TODO 1: two nested loops. Unroll both completely (#pragma unroll): the compiler can
//         keep C_r in registers only if every index into it is a constant.
static __device__ __forceinline__ void clear(float C_r[][tN], int m, int n) {
}

// Fig. 15.5. The whole block copies a height x width tile into shared memory.
//   T, lda:          the tile's first element in the matrix, and the matrix's leading dimension
//   maxRow, maxCol:  how many rows and columns of the matrix remain from that corner
//   T_s, ldas:       the shared tile, and its leading dimension
// TODO 2: the block has fewer threads than the tile has elements, so cut the tile into
//         sub-tiles of NUM_THREADS elements and copy one sub-tile per iteration.
//         Which row and column does thread threadIdx.x copy in sub-tile s? Elements
//         outside the matrix must be stored as 0.0f: then mm needs no bounds checks.
static __device__ __forceinline__ void loadTile(const float* T, int lda, int maxRow, int maxCol,
                                                float* T_s, int ldas, int height, int width) {
}

// Fig. 15.6. Add one tile pair's contribution to the thread's m x n tile c.
//   a, lda:  the thread's first row of A_s, and A_s's leading dimension
//   b, ldb:  the thread's first column of B_s, and B_s's leading dimension
//   k:       the inner dimension of the tile pair
// TODO 3: three nested loops, row, col, i, each fully unrolled.
static __device__ __forceinline__ void mm(int m, int n, int k, const float* a, int lda,
                                          const float* b, int ldb, float c[][tN]) {
}

// Fig. 15.7. Store the thread's m x n tile at c (leading dimension ldc), skipping any
// element at or beyond maxRow rows or maxCol columns.
// TODO 4: two unrolled loops and one bounds check.
static __device__ __forceinline__ void writeTile(float* c, int ldc, int maxRow, int maxCol,
                                                 float C_r[][tN], int m, int n) {
}

__global__ void __launch_bounds__(NUM_THREADS)
gemmCoarseKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // TODO 5: declare the two shared tiles, A_s (bM x bK) and B_s (bK x bN), as 1D arrays.

    // TODO 6: where this block's tile starts in C (bRow, bCol), and where this thread's
    //         tM x tN patch starts inside it (tRow, tCol). Thread 0 owns the top-left
    //         patch, thread 1 the one to its right, and so on along each row of patches.

    // TODO 7: the thread's output tile, float C_r[tM][tN], zeroed with clear().

    // TODO 8: loop over the input tile pairs, rounding up so a partial last pair counts:
    //   a. load the A tile and the B tile (loadTile, twice). Where does each start in
    //      its matrix, and how many rows and columns of the matrix remain from there?
    //   b. barrier
    //   c. mm on this thread's rows of A_s and columns of B_s
    //   d. barrier

    // TODO 9: store the patch with writeTile. A thread whose patch starts outside C must
    //         store nothing (what maxRow or maxCol gives it that?).
}

void launchGemmCoarse(const float* A, const float* B, float* C, int M, int K, int N) {
    // TODO 10: one block of NUM_THREADS threads (one-dimensional) per bM x bN tile of C;
    //          the grid's x covers the columns of C, its y the rows. Round up.
}
