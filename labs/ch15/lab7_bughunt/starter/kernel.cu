// Lab 15.7 starter: a fast linear layer in the project's layout, lab 15.6's kernel
// (milestone M3): Y[batch x out] = X[batch x in] * W^T, W stored (out, in) row-major.
//
// This kernel has THREE independent defects. It compiles, and some shapes pass. Find and
// fix all three. Run it on the CPU emulator first: its thread schedules are
// deterministic, so a synchronisation defect fails every time there instead of now and
// then, and AddressSanitizer names the line of an access outside an allocation.

#include <cstdio>

#define bM 128          // block-level output tile: bM x bN (rows of Y = batch, columns = out)
#define bN 128
#define bK 8            // depth of the input tiles in shared memory (along `in`)
#define LDA_S (bK + 1)  // leading dimension of A_s (the X tile): bK plus one padding column
#define LDB_S (bN + 4)  // pitch of B_s (the W tile, stored transposed): ours, see the header
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

// Lab 15.5's loadTile, unchanged: Fig. 15.5 with vector loads (Exercise 1). Here it loads
// the X tile. Thread t takes the float4 slot at row t / (width / 4), column
// 4 * (t % (width / 4)) of each sub-tile; one float4 load when lda is a multiple of 4 and
// the four columns are inside the matrix, four scalar loads with zero fill otherwise.
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

// The one new helper: loadTile for the W tile, which is read the way W is stored and
// stored transposed. Element (row, col) of the height x width tile at T (here row = an
// output feature n, col = an input k) goes to T_s[col * ldas + row]. The same float4
// slots and the same fallback as loadTile; only the shared-memory index differs.
static __device__ __forceinline__ void loadTileT(const float* T, int lda, int maxRow, int maxCol,
                                                 float* T_s, int ldas, int height, int width) {
    const int vecPerRow = width / 4;
    const int rowsPerSubTile = NUM_THREADS / vecPerRow;
    const int numSubTiles = height / rowsPerSubTile;
    const bool aligned = (lda % 4 == 0);
    #pragma unroll
    for (int subTile = 0; subTile < numSubTiles; ++subTile) {
        int row = subTile * rowsPerSubTile + threadIdx.x / vecPerRow;
        int col = (threadIdx.x % vecPerRow) * 4;
        float* dst = &T_s[col * ldas + row];   // column `col` of the tile is row `col` of T_s
        if (aligned && row < maxRow && col + 3 < maxCol) {
            float4 v = *(const float4*)&T[row * lda + col];
            dst[0 * ldas] = v.x;
            dst[1 * ldas] = v.y;
            dst[2 * ldas] = v.z;
            dst[3 * ldas] = v.w;
        } else {
            #pragma unroll
            for (int j = 0; j < 4; ++j) {
                dst[j * ldas] = (row < maxRow && col + j < maxCol) ? T[row * lda + col + j] : 0.0f;
            }
        }
    }
}

// Lab 15.5's mm, unchanged: rank-1 updates over the four sub-tiles. a points at the first
// row of the thread's top sub-tiles in A_s (the bottom ones start QR rows further down);
// b at the first column of its left sub-tiles in B_s (the right ones QC columns further).
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

// Lab 15.5's writeTile: c is the first element of the thread's top-left sub-tile in Y;
// maxRow and maxCol count the rows and columns of Y from there. A sub-tile row is one
// float4 store when ldc is a multiple of 4 and all four columns are inside Y, scalar
// stores otherwise; a row past maxRow is not stored at all.
template <int QR, int QC>
static __device__ __forceinline__ void writeTile(float* c, int ldc, int maxRow, int maxCol, float C_r[][tN]) {
    const bool vec = (ldc % 4 == 0);
    #pragma unroll
    for (int qr = 0; qr < 2; ++qr) {
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            const int row = qr * QR + r;
            #pragma unroll
            for (int qc = 0; qc < 2; ++qc) {
                const int col = qc * QC;
                const int cr = qr * 4 + r, cc = qc * 4;   // the same row in C_r
                if (vec && col + 3 < maxCol) {
                    *(float4*)&c[row * ldc + col] =
                        make_float4(C_r[cr][cc], C_r[cr][cc + 1], C_r[cr][cc + 2], C_r[cr][cc + 3]);
                } else if (row < maxRow) {
                    #pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        if (col + j < maxCol) c[row * ldc + col + j] = C_r[cr][cc + j];
                    }
                }
            }
        }
    }
}

// One block computes a 128 x 128 tile of Y; blockIdx.x walks the outputs, blockIdx.y the
// batch. WARP is the warp width section 15.5's layout is built for (32 or 64).
template <int WARP>
__global__ void __launch_bounds__(NUM_THREADS)
fastLinearKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    // Two pairs of tiles: pair `tile` lives in buffers tile & 1.
    __shared__ float A_s[2][bM * LDA_S];   // X tile: A_s[row][k], padded
    __shared__ float B_s[2][bK * LDB_S];   // W tile, transposed: B_s[k][n]

    const int bRow = blockIdx.y * bM;      // first row of Y (batch element) of this block
    const int bCol = blockIdx.x * bN;      // first column of Y (output feature)

    // Section 15.5's layout for a warp of WARP threads (ch15-facts B1 for 64).
    constexpr int LC = WARP / 8;           // lanes per lane-row
    constexpr int QR = (WARP / LC) * 4;    // quadrant height: 8 lane-rows of 4 rows = 32
    constexpr int QC = LC * 4;             // quadrant width
    constexpr int WC = bN / (2 * QC);      // warps across the block tile
    const int warp = threadIdx.x / WARP;
    const int lane = threadIdx.x % WARP;
    const int tRow = (warp / WC) * (2 * QR) + (lane / LC) * 4;   // the thread's top-left sub-tile
    const int tCol = (warp % WC) * (2 * QC) + (lane % LC) * 4;

    float C_r[tM][tN];
    clear(C_r, tM, tN);

    const int numTiles = (in + bK - 1) / bK;
    if (numTiles > 0) {   // in == 0: nothing to add, and no buffer was loaded
        // Pair 0 into buffers 0. The W tile is rows bCol.. of W, columns 0.. of `in`.
        loadTile(&X[(size_t)bRow * in], in, batch - bRow, in, A_s[0], LDA_S, bM, bK);
        loadTileT(&W[(size_t)bCol * in], in, out - bCol, in, B_s[0], LDB_S, bN, bK);
        __syncthreads();

        for (int tile = 0; tile < numTiles - 1; ++tile) {
            const int cur = tile & 1, next = (tile + 1) & 1;
            mm<QR, QC>(bK, &A_s[cur][tRow * LDA_S], bK, &B_s[cur][tCol], LDB_S, C_r);
            // No barrier here: pair tile + 1 goes into the other buffers.
            const int k0 = (tile + 1) * bK;
            loadTile(&X[(size_t)bRow * in + k0], in, batch - bRow, in - k0, A_s[next], LDA_S, bM, bK);
            loadTileT(&W[(size_t)bCol * in + k0], in, out - bCol, in - k0, B_s[next], LDB_S, bN, bK);
        }

        const int last = (numTiles - 1) & 1;
        mm<QR, QC>(bK, &A_s[last][tRow * LDA_S], bK, &B_s[last][tCol], LDB_S, C_r);
    }

    float* y = &Y[(size_t)(bRow + tRow) * out + bCol + tCol];
    const int maxRow = (bRow + tRow < batch) ? (batch - (bRow + tRow)) : 0;
    const int maxCol = (bCol + tCol < out) ? (out - (bCol + tCol)) : 0;
    writeTile<QR, QC>(y, out, maxRow, maxCol, C_r);
}

// For the tests: the layout for a warp of `width` threads, on any device. The width only
// changes which thread owns which outputs, so both layouts must give the same Y.
void launchLinearWidth(const float* X, const float* W, float* Y, int batch, int in, int out, int width) {
    if (batch <= 0 || out <= 0) return;
    dim3 grid((out + bN - 1) / bN,       // x covers the outputs (columns of Y)
              (batch + bM - 1) / bM);    // y covers the batch (rows of Y)
    switch (width) {
        case 32: fastLinearKernel<32><<<grid, NUM_THREADS>>>(X, W, Y, batch, in, out); break;
        case 64: fastLinearKernel<64><<<grid, NUM_THREADS>>>(X, W, Y, batch, in, out); break;
        default: std::fprintf(stderr, "launchLinearWidth: no layout for a warp of %d threads\n", width);
    }
}

// Milestone M3, lab 6.4's signature: the layout for this device's warp width, read once
// from device 0's properties (32 on NVIDIA; 64 or 32 on AMD, ch15-facts B1).
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out) {
    static int width = 0;
    if (width == 0) {
        cudaDeviceProp prop;
        width = (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) ? prop.warpSize : -1;
    }
    launchLinearWidth(X, W, Y, batch, in, out, width);
}
