// Lab 19.1 starter (CUDA; also built for HIP and for the CPU emulators).
// One convolutional layer, the book's (stride 1, no padding), computed two ways:
//   X: N x C x H x W, F: M x C x K x K, Y: N x M x H_out x W_out, row-major float,
//   H_out = H - K + 1, W_out = W - K + 1, P = H_out * W_out.
// Book: section 19.2 (pp. 460-463, Figs. 19.5 and 19.7) and section 19.3 (pp. 463-472).
// The build defines TILE_WIDTH (16), so do not hard-code 16.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// The project's M2 GEMM, lab 6.4 (provided/gemm.cmake picks which copy):
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out);

// ---------------------------------------------------------------- (1) direct
// Fig. 19.7 (p. 462), one thread per output pixel, launched as in Fig. 19.5 (p. 461):
// blocks of TILE_WIDTH x TILE_WIDTH threads, grid (M, T, N), where T is the number of
// tiles in one output map and blockIdx.y numbers them row by row, W_grid to a row.
// The page writes X[n, c, h + p, w + q]: you linearise every index by hand (row-major,
// highest dimension first). The kernel takes H, W and M, which the page's does not.
__global__ void convDirectKernel(const float* X, const float* F, float* Y,
                                 int C, int H, int W, int M, int K, int W_grid) {
    // TODO 1: which output map m, row h, column w and sample n does this thread own?
    //         blockIdx.y holds a tile number: recover the tile's row and column from it.
    // TODO 2: LeNet C1's output is 28 x 28. With 16 x 16 tiles some threads have no
    //         pixel. What must they not do?
    // TODO 3: sum X[n, c, h + p, w + q] * F[m, c, p, q] over c, p, q in a register,
    //         then write Y[n, m, h, w] once.
}

void launchConvDirect(const float* X, const float* F, float* Y,
                      int N, int C, int H, int W, int M, int K) {
    // TODO 4: Fig. 19.5's W_grid, H_grid and T, but so that every pixel is covered
    //         when the output side is not a multiple of TILE_WIDTH (the page's integer
    //         division is not). Then launch convDirectKernel. Name the dim3 variables
    //         dimBlock and dimGrid: blockDim and gridDim are built-in names.
}

// ---------------------------------------------------------------- (2) unroll + GEMM
// The unrolled input of sample n, stored transposed: Bt = B^T has P rows, one per output
// pixel v (row-major over h, w), and C*K*K columns u holding that pixel's patches in the
// book's (c, p, q) order (pp. 465-466). Element (v, u) of Bt is element (u, v) of B.
// Eq. 19.5 (p. 470) says where B[u, v] comes from in X. Bt holds exactly P * C*K*K floats.
// The book gives no kernel for this step; how you map threads to elements is yours.
__global__ void unrollKernel(const float* X, float* Bt, int C, int H, int W, int K) {
    // TODO 5: fill Bt for one sample. Decide which element(s) each thread writes, check
    //         that they exist, and fetch each from X with Eq. 19.5.
    //         Which mapping writes Bt contiguously? Which reads X contiguously?
}

void launchUnroll(const float* X, int n, float* Bt, int C, int H, int W, int K) {
    // TODO 6: launch unrollKernel over sample n's P * C*K*K elements. X is the whole
    //         batch: either give the kernel sample n's first float, or n itself.
}

void launchConvGemm(const float* X, const float* F, float* Y, float* Bt,
                    int N, int C, int H, int W, int M, int K) {
    // TODO 7: for each sample n, unroll it into Bt, then call launchLinear ONCE so that
    //         sample n's Y comes out as M rows of P (NCHW), without copying F.
    //         launchLinear computes Y = X * W^T with W stored (out, in). F is already an
    //         M x C*K*K matrix (p. 465), Bt is P x C*K*K, and you want F * B.
    //         Which of them plays X, which plays W, what are batch, in and out, and where
    //         does sample n's Y start?
}
