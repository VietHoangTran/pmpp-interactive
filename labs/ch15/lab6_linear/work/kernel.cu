// Lab 15.6 starter (CUDA; also built for HIP and for the CPU emulator).
// Milestone M3: the project's linear layer, fast.
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major,
// with lab 6.4's launchLinear signature, so it drops in where M2 was.
//
// START FROM YOUR LAB 15.5 KERNEL. In the book's names A = X (M = batch, K = in),
// B = W^T (N = out), C = Y. What changes (README, "Beyond the book"):
// 1. B = W^T is stored column-major: W's rows are B's columns. Load the W tile the way W
//    is stored, contiguous along k, and store it transposed into B_s[k][n] (loadTileT
//    below, the one new helper), so that mm still reads a B strip as one contiguous row
//    of B_s.
// 2. B_s gets a pitch of its own, LDB_S.
// 3. The kernel's template parameter is WARP, not W: here W is the weight matrix.
// launchLinearWidth and launchLinear at the bottom are complete.

#include <cstdio>

#define bM 128          // block-level output tile: bM x bN (rows of Y = batch, columns = out)
#define bN 128
#define bK 8            // depth of the input tiles in shared memory (along `in`)
#define LDA_S (bK + 1)  // leading dimension of A_s (the X tile): bK plus one padding column
// TODO: the pitch of B_s, the W tile stored transposed. bN gives correct results, but the
// transposed store then conflicts on 32 banks. Section 15.6's explorer, third mode, finds
// a pitch that does not.
#define LDB_S bN
#define tM 8            // thread-level output tile (logical): tM x tN
#define tN 8
#define NUM_THREADS 256 // the book's NUM_THREADS_PER_BLOCK

static __device__ __forceinline__ void clear(float C_r[][tN], int m, int n) {
    // TODO: your lab 15.5 clear (Fig. 15.4).
}

// Loads the X tile, exactly as lab 15.5's loadTile loaded the A tile.
static __device__ __forceinline__ void loadTile(const float* T, int lda, int maxRow, int maxCol,
                                                float* T_s, int ldas, int height, int width) {
    // TODO: your lab 15.5 loadTile.
}

// New: loads the W tile. The tile is height rows (output features n) by width columns
// (inputs k) of the row-major matrix at T, whose rows are lda floats apart; maxRow and
// maxCol count the rows and columns of the matrix from T. Element (row, col) goes to
// T_s[col * ldas + row]: stored transposed. Read it the way loadTile reads, one float4
// per thread when lda is a multiple of 4 and the four floats are inside the matrix,
// scalars with zero fill otherwise.
static __device__ __forceinline__ void loadTileT(const float* T, int lda, int maxRow, int maxCol,
                                                 float* T_s, int ldas, int height, int width) {
    // TODO
}

template <int QR, int QC>
static __device__ __forceinline__ void mm(int k, const float* a, int lda, const float* b, int ldb,
                                          float c[][tN]) {
    // TODO: your lab 15.5 mm.
}

template <int QR, int QC>
static __device__ __forceinline__ void writeTile(float* c, int ldc, int maxRow, int maxCol, float C_r[][tN]) {
    // TODO: your lab 15.5 writeTile.
}

// One block computes a 128 x 128 tile of Y; blockIdx.x walks the outputs, blockIdx.y the
// batch. WARP is the warp width section 15.5's layout is built for (32 or 64).
template <int WARP>
__global__ void __launch_bounds__(NUM_THREADS)
fastLinearKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    // TODO: your lab 15.5 kernel body, with the X tile where A was, the W tile (through
    // loadTileT) where B was, and B_s sized and indexed with LDB_S. Which rows of W does
    // a block need, and where along k does tile `tile` start?
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
