// Lab 15.3 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Coalesced stores of the output tile: lab 15.2's kernel with the threads re-mapped so
// that each owns four 4 x 4 sub-tiles of C instead of one 8 x 8 patch, and each row of a
// sub-tile is stored with one float4 store.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.5 (pp. 360-361), Fig. 15.10; Exercises 3 and 2 (p. 370). The book
// gives no code for this section. Milestone M3, stage 3 of 5.
//
// The layout, for a warp of W threads (the book's numbers are the W = 32 case):
//   lanes:      a warp's W lanes form 8 rows of LC = W / 8 lanes (8 x 4 when W = 32);
//               lane l sits at row l / LC, column l % LC.
//   quadrants:  each warp's tile is split into 2 x 2 quadrants of QR x QC elements,
//               QR = 8 lane-rows x 4 = 32, QC = LC x 4 (32 x 16 when W = 32);
//               a lane owns the 4 x 4 sub-tile at the same place in each quadrant.
//   warp tile:  2 QR x 2 QC = 64 x 8 LC (64 x 32 when W = 32);
//               WC = 128 / (8 LC) warps across the block tile (2 x 4 warps when W = 32).
// With W = 64 (AMD CDNA) the same rules give 8 x 8 lanes, 32 x 32 quadrants, 64 x 64
// warp tiles and 2 x 2 warps (notes/today/ch15-facts.md B1). The thread's four sub-tiles
// are its logical 8 x 8 tile C_r: element (qr * 4 + r, qc * 4 + c) is row r, column c of
// the sub-tile in quadrant (qr, qc).
//
// Why a template: the layout must be known at compile time (the loops are unrolled and
// C_r must stay in registers), but the warp width is known only on the host: on HIP,
// warpSize "can not be assumed to be a compile-time constant on the host" (B1). AMD's
// recommended way is the one used here: the host reads the width and launches the
// matching instance. The kernel never assumes a width.
//
// Where this file differs from the page, beyond lab 15.1's list (1D block, tiles at the
// top, int sizes and lab 5.3's argument order, static helpers, __launch_bounds__):
// - The layout is our implementation of the section's description; the book leaves it
//   to Exercise 3.
// - The float4 store is the book's idiom from chapter 6: cast the float pointer to a
//   float4 pointer. No vendor reference documents that cast; NVIDIA's developer blog
//   requires the address to be 16-byte aligned (ch06-facts.md A11). On CUDA, C comes from
//   cudaMalloc (aligned to at least 256 bytes, ch06-facts.md A3), sub-tile columns are
//   multiples of 4, so every row's address is aligned when N is a multiple of 4. When it
//   is not, or when the row's four columns do not all lie inside C, the stores are
//   scalar. On HIP, C comes from hipMalloc, and no facts file documents its alignment:
//   that the same rule holds there is unverified.
#include <cstdio>

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in shared memory
#define tM 8            // thread-level output tile (logical): tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

// Fig. 15.4, unchanged.
static __device__ __forceinline__ void clear(float C_r[][tN], int m, int n) {
    #pragma unroll
    for (int row = 0; row < m; ++row) {
        #pragma unroll
        for (int col = 0; col < n; ++col) {
            C_r[row][col] = 0.0f;
        }
    }
}

// Fig. 15.5, unchanged.
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

// Lab 15.2's rank-1 updates, over the four sub-tiles. a points at the first row of the
// thread's top sub-tiles in A_s (the bottom ones start QR rows further down); b at the
// first column of its left sub-tiles in B_s (the right ones start QC columns further on).
// Per i the thread loads two 4-value pieces of A and two of B, one scalar shared read
// each, so the documented 32-bit bank rule applies to every read (ch15-facts.md A2).
template <int QR, int QC>
static __device__ __forceinline__ void mm(int k, const float* a, int lda, const float* b, int ldb,
                                          float c[][tN]) {
    #pragma unroll
    for (int i = 0; i < k; ++i) {
        float a_r[tM];   // rows 0-3 of the top sub-tiles, then rows 0-3 of the bottom ones
        float b_r[tN];   // columns 0-3 of the left sub-tiles, then of the right ones
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            a_r[r] = a[r * lda + i];
            a_r[4 + r] = a[(QR + r) * lda + i];
        }
        #pragma unroll
        for (int col = 0; col < 4; ++col) {
            b_r[col] = b[i * ldb + col];
            b_r[4 + col] = b[i * ldb + QC + col];
        }
        #pragma unroll
        for (int row = 0; row < tM; ++row) {
            #pragma unroll
            for (int col = 0; col < tN; ++col) {
                c[row][col] += a_r[row] * b_r[col];
            }
        }
    }
}

// Store the four sub-tiles. c is the first element of the top-left sub-tile in C;
// maxRow and maxCol count the rows and columns of C from there (Fig. 15.7's contract).
// A row of a sub-tile is four consecutive floats of C: one float4 store when the row's
// address is 16-byte aligned (ldc a multiple of 4) and all four columns are inside C,
// scalar stores with bounds checks otherwise.
template <int QR, int QC>
static __device__ __forceinline__ void writeTile(float* c, int ldc, int maxRow, int maxCol, float C_r[][tN]) {
    const bool vec = (ldc % 4 == 0);
    #pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = qr * QR + r;
            if (row >= maxRow) continue;
            #pragma unroll
            for (int qc = 0; qc < 2; ++qc) {
                const int col = qc * QC;
                const int cr = qr * 4 + r, cc = qc * 4;   // the same row in C_r
                if (vec && col + 3 < maxCol) {
                    *(float4*)&c[row * ldc + col] =
                        make_float4(C_r[cr][cc], C_r[cr][cc + 1], C_r[cr][cc + 2], C_r[cr][cc + 3]);
                } else {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (col + j < maxCol) c[row * ldc + col + j] = C_r[cr][cc + j];
                    }
                }
            }
        }
    }
}

template <int W>
__global__ void __launch_bounds__(NUM_THREADS)
gemmRemapKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float A_s[bM * bK];
    __shared__ float B_s[bK * bN];

    const int bRow = blockIdx.y * bM;
    const int bCol = blockIdx.x * bN;

    // Section 15.5's layout for a warp of W threads (see the header).
    constexpr int LC = W / 8;          // lanes per lane-row
    constexpr int QR = (W / LC) * 4;   // quadrant height: 8 lane-rows of 4 rows = 32
    constexpr int QC = LC * 4;         // quadrant width
    constexpr int WC = bN / (2 * QC);  // warps across the block tile
    const int warp = threadIdx.x / W;
    const int lane = threadIdx.x % W;
    const int lRow = (lane / LC) * 4;  // the lane's sub-tile inside a quadrant
    const int lCol = (lane % LC) * 4;
    const int wRow = (warp / WC) * (2 * QR);   // the warp's tile inside the block tile
    const int wCol = (warp % WC) * (2 * QC);
    const int tRow = wRow + lRow;      // the thread's top-left sub-tile
    const int tCol = wCol + lCol;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    for (int tile = 0; tile < (K + bK - 1) / bK; ++tile) {
        loadTile(&A[bRow * K + tile * bK], K, M - bRow, K - tile * bK, &A_s[0], bK, bM, bK);
        loadTile(&B[tile * bK * N + bCol], N, K - tile * bK, N - bCol, &B_s[0], bN, bK, bN);
        __syncthreads();
        mm<QR, QC>(bK, &A_s[tRow * bK], bK, &B_s[tCol], bN, C_r);
        __syncthreads();
    }

    float* c = &C[(bRow + tRow) * N + bCol + tCol];
    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile<QR, QC>(c, N, maxRow, maxCol, C_r);
}

// Launch the layout for a warp of `width` threads. The kernel is compiled for the widths
// GPUs have: 32 (NVIDIA, and AMD RDNA) and 64 (AMD CDNA) (ch15-facts.md B1). The tests
// call this directly to check both layouts on any backend.
void launchGemmRemapWidth(const float* A, const float* B, float* C, int M, int K, int N, int width) {
    dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM);
    switch (width) {
        case 32: gemmRemapKernel<32><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        case 64: gemmRemapKernel<64><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        default: std::fprintf(stderr, "launchGemmRemapWidth: no layout for a warp of %d threads\n", width);
    }
}

// Launch the layout for this device's warp width, read once from the device's properties
// (device 0, the device the lab runtime uses).
void launchGemmRemap(const float* A, const float* B, float* C, int M, int K, int N) {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    launchGemmRemapWidth(A, B, C, M, K, N, width);
}
