// Lab 15.5 starter (CUDA; also built for HIP and for the CPU emulator).
// Double buffering and software pipelining. This is lab 15.4's solution, renamed, with its
// tile loop taken out. Put it back so that iteration `tile` computes with pair `tile`
// while it loads pair tile + 1 into a second set of buffers, with ONE barrier per
// iteration instead of two.
//   C[M x N] = A[M x K] * B[K x N], row-major, any sizes.
// Book: section 15.8 (pp. 364-368), Figs. 15.12 to 15.14. Milestone M3, stage 5 of 5.
//
// Before you write it, decide which of lab 15.4's two barriers is a true dependence and
// which a false one (Fig. 15.12), and why a thread that runs ahead can never overwrite a
// pair its neighbours are still reading. The tests launch every case 10 times on a GPU;
// the emulators catch a missing or misplaced barrier every time.

#include <cstdio>

#define bM 128          // block-level output tile: bM x bN
#define bN 128
#define bK 8            // depth of the input tiles in shared memory
#define LDA_S (bK + 1)  // leading dimension of A_s: bK plus one padding column
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

// Fig. 15.5 with vector loads (Exercise 1). The tile is cut into float4s, width / 4 per
// row; thread t copies the float4 at row t / (width / 4), column 4 * (t % (width / 4)) of
// each sub-tile of NUM_THREADS float4s. For both of this kernel's tiles that is one
// sub-tile: every thread loads one float4 per tile.
// When lda is a multiple of 4 (so the float4 is 16-byte aligned, see the header) and all
// four columns are inside the matrix, one float4 load; otherwise four scalar loads, with
// zeros past the matrix as before. The shared stores are scalar in both cases.
static __device__ __forceinline__ void loadTile(const float* T, int lda, int maxRow, int maxCol,
                                                float* T_s, int ldas, int height, int width) {
    const int vecPerRow = width / 4;
    const int rowsPerSubTile = NUM_THREADS / vecPerRow;
    const int numSubTiles = height / rowsPerSubTile;
    const bool aligned = (lda % 4 == 0);
    #pragma unroll
    for (int subTile = 0; subTile < numSubTiles; ++subTile) {
        int row = subTile * rowsPerSubTile + threadIdx.x / vecPerRow;
        int col = (threadIdx.x % vecPerRow) * 4;
        float* dst = &T_s[row * ldas + col];
        if (aligned && row < maxRow && col + 3 < maxCol) {
            float4 v = *(const float4*)&T[row * lda + col];
            dst[0] = v.x;
            dst[1] = v.y;
            dst[2] = v.z;
            dst[3] = v.w;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst[j] = (row < maxRow && col + j < maxCol) ? T[row * lda + col + j] : 0.0f;
            }
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
gemmPipelinedKernel(const float* A, const float* B, float* C, int M, int K, int N) {
    // TODO 1: two pairs of tiles instead of one.
    __shared__ float A_s[bM * LDA_S];
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

    // TODO 2: the tile loop of Fig. 15.14, written with this file's helpers. Its shape:
    //   prologue:  load pair 0 (an A tile and a B tile) into the first buffers; barrier
    //   loop over tile = 0 .. numTiles - 2:
    //              mm on pair `tile`
    //              (no barrier)
    //              load pair tile + 1 into the other buffers
    //              barrier
    //   epilogue:  mm on the last pair
    // The book swaps a "current" and a "next" pointer at the end of each iteration; you
    // may instead pick a pair's buffers by its parity, tile & 1, as lab 6.5 does.
    // The figure leaves two cases for you: numTiles - 1 when numTiles is 0 (K = 0), and
    // which buffers the epilogue reads when numTiles is odd and when it is even.

    float* c = &C[(bRow + tRow) * N + bCol + tCol];
    const int maxRow = (bRow + tRow < M) ? (M - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < N) ? (N - (bCol + tCol)) : 0;
    writeTile<QR, QC>(c, N, maxRow, maxCol, C_r);
}

// Launch the layout for a warp of `width` threads. The kernel is compiled for the widths
// GPUs have: 32 (NVIDIA, and AMD RDNA) and 64 (AMD CDNA) (ch15-facts.md B1). The tests
// call this directly to check both layouts on any backend.
void launchGemmPipelinedWidth(const float* A, const float* B, float* C, int M, int K, int N, int width) {
    dim3 grid((N + bN - 1) / bN, (M + bM - 1) / bM);
    switch (width) {
        case 32: gemmPipelinedKernel<32><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        case 64: gemmPipelinedKernel<64><<<grid, NUM_THREADS>>>(A, B, C, M, K, N); break;
        default: std::fprintf(stderr, "launchGemmPipelinedWidth: no layout for a warp of %d threads\n", width);
    }
}

// Launch the layout for this device's warp width, read once from the device's properties
// (device 0, the device the lab runtime uses).
void launchGemmPipelined(const float* A, const float* B, float* C, int M, int K, int N) {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    launchGemmPipelinedWidth(A, B, C, M, K, N, width);
}
