// Lab 15.4 starter (CUDA; also built for HIP and for the CPU emulator).
// Padding and vector loads. This is lab 15.3's solution, renamed. Two changes:
//   1. Give A_s one padding column (section 15.6): its leading dimension becomes bK + 1.
//   2. Rewrite loadTile so that each thread fetches four floats with one float4 load
//      (Exercise 1), wherever that is safe.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.6 (pp. 361-363), Fig. 15.11; Exercise 1 (p. 370); the low-occupancy
// argument for vector loads (section 15.7, p. 364). Milestone M3, stage 4 of 5.
//
// The book changes the declaration and the leading dimension passed to loadTile and mm,
// and stops there. Find every place this file indexes A_s, not only those.
// Give the padded leading dimension a name (a #define) and use it in every A_s index, so
// that no index can keep the unpadded width.

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

// Fig. 15.5 with vector loads (Exercise 1). Same contract as before: copy a height x width
// tile into shared memory, zeros for elements outside the matrix.
// TODO 2: cut the tile into float4s (four consecutive floats of a row) and give each
//   thread one float4 per sub-tile; for both of this kernel's tiles, how many sub-tiles
//   is that? When may a thread load its four floats with one float4 load (the chapter 6
//   idiom: cast the address to const float4*)? Think about alignment (A and B come from
//   cudaMalloc, aligned to at least 256 bytes; what makes the address of a tile row
//   16-byte aligned?) and about columns past the matrix. Otherwise fall back to four
//   scalar loads with zero fill. Store into shared memory one float at a time: why can a
//   padded row of A_s not take a float4 store?
static __device__ __forceinline__ void loadTile(const float* T, int lda, int maxRow, int maxCol,
                                                float* T_s, int ldas, int height, int width) {
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
gemmPaddedKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    __shared__ float A_s[bM * bK];   // TODO 1: one padding column
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
void launchGemmPaddedWidth(const float* A, const float* B, float* C, int M, int K, int N, int width) {
    dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM);
    switch (width) {
        case 32: gemmPaddedKernel<32><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        case 64: gemmPaddedKernel<64><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        default: std::fprintf(stderr, "launchGemmPaddedWidth: no layout for a warp of %d threads\n", width);
    }
}

// Launch the layout for this device's warp width, read once from the device's properties
// (device 0, the device the lab runtime uses).
void launchGemmPadded(const float* A, const float* B, float* C, int M, int K, int N) {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    launchGemmPaddedWidth(A, B, C, M, K, N, width);
}
