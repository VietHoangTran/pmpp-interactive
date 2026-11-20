// Lab 15.3 starter (CUDA; also built for HIP and for the CPU emulator).
// Coalesced stores of the output tile. This is lab 15.2's solution, renamed and made a
// template over the warp width W. Re-map the threads so that each owns four 4 x 4
// sub-tiles of C instead of one 8 x 8 patch, and store each row of a sub-tile with one
// float4 store.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.5 (pp. 360-361), Fig. 15.10; Exercises 3 and 2 (p. 370). The book
// gives no code for this section. Milestone M3, stage 3 of 5.
//
// The layout, for a warp of W threads (the book describes the W = 32 case):
//   - a warp's W lanes form 8 lane-rows of LC = W / 8 lanes;
//   - the warp's tile is split into 2 x 2 quadrants, each 32 rows by 4 * LC columns;
//     a lane owns the 4 x 4 sub-tile at the same place in all four quadrants;
//   - so a warp's tile is 64 x 8 * LC, and the block tile holds 128 / (8 * LC) warps
//     across and as many rows of warps as the block's 256 threads make.
// Work out where lane l's sub-tile sits in a quadrant, and where warp w's tile sits in
// the block tile, before you write any code. Check your answer for W = 32 against the
// book: 8 warps as 2 x 4, warp tiles of 64 x 32, quadrants of 32 x 16.
//
// W is a template parameter because the loops are unrolled and the tile must stay in
// registers, so the layout has to be known when the kernel is compiled; the warp width
// is known only on the host (on HIP it differs between devices). Never write the width
// as a number inside the kernel.

#include <cstdio>

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in shared memory
#define tM 8            // thread-level output tile (logical): tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

static __device__ __forceinline__ void clear(float C_r[][tN], int m, int n) {
    #pragma unroll
    for (int row = 0; row < m; ++row) {
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

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

// TODO 2 (mm): this is lab 15.2's mm, which reads a contiguous 8 x 8 patch: rows
//   0..7 of a and columns 0..7 of b. The thread's rows of A_s now come as two groups of
//   4 (the top and the bottom quadrants), and its columns of B_s as two groups of 4 (the
//   left and the right quadrants). Change how a_r and b_r are filled so that a_r holds
//   the thread's 4 + 4 rows and b_r its 4 + 4 columns; the rank-1 update stays as it is.
//   Keep every shared read a scalar read. You will need the quadrant height and width
//   here: pass them as template parameters (they must be constants).
static __device__ __forceinline__ void mm(int m, int n, int k, const float* a, int lda,
                                          const float* b, int ldb, float c[][tN]) {
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        float a_r[tM];
        #pragma unroll
        for (int row = 0; row < m; ++row) {
            a_r[row] = a[row * lda + i];
        }
        float b_r[tN];
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            b_r[col] = b[i * ldb + col];
        }
        #pragma unroll
        for (int row = 0; row < m; ++row) {
            #pragma unroll
            for (int col = 0; col < n; ++col) {
                c[row][col] += a_r[row] * b_r[col];
            }
        }
    }
}

// TODO 3 (writeTile): store the thread's four 4 x 4 sub-tiles. Keep Fig. 15.7's contract:
//   c is the first element of the top-left sub-tile in C, and maxRow, maxCol count the
//   rows and columns of C from there. A row of a sub-tile is four consecutive floats of
//   C. Store it with ONE float4 store (the chapter 6 idiom: cast the address to float4*)
//   when that is safe, and with four bounds-checked scalar stores otherwise.
//   When is it safe? A float4 store needs a 16-byte-aligned address and all four columns
//   inside C. (C itself comes from cudaMalloc, which is aligned to at least 256 bytes.)
//   The tests include N = 131 and N = 7: think about those before you run them.
static __device__ __forceinline__ void writeTile(float* c, int ldc, int maxRow, int maxCol,
                                                 float C_r[][tN], int m, int n) {
}

template <int W>
__global__ void __launch_bounds__(NUM_THREADS)
gemmRemapKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float A_s[bM * bK];
    __shared__ float B_s[bK * bN];

    const int bRow = blockIdx.y * bM;
    const int bCol = blockIdx.x * bN;

    // TODO 1 (mapping): this is lab 15.2's mapping, one 8 x 8 patch per thread in row-major
    //   order. Replace it with section 15.5's, for a warp of W threads: the thread's warp
    //   and lane, the lane's sub-tile inside a quadrant, the warp's tile inside the block
    //   tile, and from those tRow and tCol, the top-left corner of the thread's top-left
    //   sub-tile. Make every size that depends on W a constexpr.
    const int tilesPerBlockX = bN / tN;
    const int ty = threadIdx.x / tilesPerBlockX;
    const int tx = threadIdx.x % tilesPerBlockX;
    const int tRow = ty * tM;
    const int tCol = tx * tN;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    for (int tile = 0; tile < (K + bK - 1) / bK; ++tile) {
        loadTile(&A[bRow * K + tile * bK], K, M - bRow, K - tile * bK, &A_s[0], bK, bM, bK);
        loadTile(&B[tile * bK * N + bCol], N, K - tile * bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();
        mm(tM, tN, bK, &A_s[tRow * bK], bK, &B_s[tCol], bN, C_r);   // TODO 2: the new mm
        __syncthreads();
    }

    float* c = &C[(bRow + tRow) * N + bCol + tCol];
    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile(c, N, maxRow, maxCol, C_r, tM, tN);                   // TODO 3: the new writeTile
}

// Given: launch gemmRemapKernel<W> for a warp of `width` threads, with lab 15.2's grid and
// block. The kernel is compiled for the two widths GPUs have: 32 (NVIDIA, and AMD RDNA)
// and 64 (AMD CDNA). The tests call this with both widths on every backend, and the
// benchmark's occupancy query refers to both instances, so it is written for you.
void launchGemmRemapWidth(const float* A, const float* B, float* C, int M, int K, int N, int width) {
    dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM);
    switch (width) {
        case 32: gemmRemapKernel<32><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        case 64: gemmRemapKernel<64><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        default: std::fprintf(stderr, "launchGemmRemapWidth: no layout for a warp of %d threads\n", width);
    }
}

// TODO 4: launch the layout for this device's warp width. The host is the only place that
//   knows it: read it once from cudaGetDeviceProperties (device 0, the one the lab runtime
//   uses), keep it for later calls, and call launchGemmRemapWidth.
void launchGemmRemap(const float* A, const float* B, float* C, int M, int K, int N) {
}
