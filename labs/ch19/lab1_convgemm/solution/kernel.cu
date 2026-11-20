// Lab 19.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
// One convolutional layer, the book's (stride 1, no padding), computed two ways:
//   X: N x C x H x W, F: M x C x K x K, Y: N x M x H_out x W_out, row-major float,
//   H_out = H - K + 1, W_out = W - K + 1.
// Book: section 19.2 (pp. 460-463, Figs. 19.5 and 19.7) and section 19.3 (pp. 463-472).
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// The project's M2 GEMM, lab 6.4 (provided/gemm.cmake picks which copy):
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out);

// [lesson:direct]
// Fig. 19.7 (p. 462) with its indices linearised by hand, and Fig. 19.5's host code
// (p. 461). One thread per output pixel; block (m, tile, n). Where it differs from the
// page: the signature also carries H, W and M, which the linearised indices need;
// W_grid and H_grid round up, so an output side that is not a multiple of TILE_WIDTH
// (LeNet C1's 28) is covered; and the thread checks that its pixel exists.
__global__ void convDirectKernel(const float* X, const float* F, float* Y,
                                 int C, int H, int W, int M, int K, int W_grid) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int m = blockIdx.x;
    int h = (blockIdx.y / W_grid) * TILE_WIDTH + threadIdx.y;   // tile row, then row in the tile
    int w = (blockIdx.y % W_grid) * TILE_WIDTH + threadIdx.x;   // tile column, then column
    int n = blockIdx.z;
    if (h < H_out && w < W_out) {   // the edge tiles hang over the map: not on the page
        float acc = 0.;
        for (int c = 0; c < C; c++) {
            for (int p = 0; p < K; p++) {
                for (int q = 0; q < K; q++) {
                    acc += X[((n * C + c) * H + (h + p)) * W + (w + q)]   // X[n, c, h + p, w + q]
                         * F[((m * C + c) * K + p) * K + q];             // F[m, c, p, q]
                }
            }
        }
        Y[((n * M + m) * H_out + h) * W_out + w] = acc;                  // Y[n, m, h, w]
    }
}

void launchConvDirect(const float* X, const float* F, float* Y,
                      int N, int C, int H, int W, int M, int K) {
    int H_out = H - K + 1, W_out = W - K + 1;
    int W_grid = (W_out + TILE_WIDTH - 1) / TILE_WIDTH;   // tiles across; the page truncates
    int H_grid = (H_out + TILE_WIDTH - 1) / TILE_WIDTH;   // tiles down
    int T = H_grid * W_grid;                              // tiles per output map
    // Named dimBlock and dimGrid, as in lab 6.4: the page's blockDim and gridDim would
    // hide the built-in variables of the same names inside this function.
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid(M, T, N);
    convDirectKernel<<<dimGrid, dimBlock>>>(X, F, Y, C, H, W, M, K, W_grid);
}
// [/lesson:direct]

// [lesson:unroll]
// Explicit unrolling (p. 467 describes it; the book gives no kernel, so this one is ours).
// It writes sample n's unrolled input transposed, as Bt = B^T: P = H_out * W_out rows,
// one per output pixel v, each holding the C*K*K values of that pixel's patches in the
// book's (c, p, q) order. Element (v, u) of Bt is element (u, v) of B, Eq. 19.5 (p. 470):
//   B[u, v] = X[n, u / K^2, (u % K^2) / K + v / W_out, (u % K^2) % K + v % W_out]
// One thread per element of Bt, numbered in Bt's own row-major order, so neighbouring
// threads write neighbouring floats. Their reads walk X along q, then jump to the next
// row of the patch: the opposite mapping (neighbouring threads on neighbouring v) would
// read X contiguously and write Bt with a stride of C*K*K. One of the two has to give
// unless the block stages the patch through shared memory.
__global__ void unrollKernel(const float* Xn, float* Bt, int C, int H, int W, int K) {
    int W_out = W - K + 1;
    int P = (H - K + 1) * W_out;
    int CKK = C * K * K;
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // element of Bt
    if (i < P * CKK) {
        int v = i / CKK, u = i % CKK;                 // row of Bt (pixel), column (patch value)
        int c = u / (K * K), r = u % (K * K);         // channel, place in the K x K patch
        int h = v / W_out + r / K;                    // patch corner plus offset
        int w = v % W_out + r % K;
        Bt[i] = Xn[(c * H + h) * W + w];              // Xn is sample n: X[n, c, h, w]
    }
}

void launchUnroll(const float* X, int n, float* Bt, int C, int H, int W, int K) {
    int P = (H - K + 1) * (W - K + 1);
    int total = P * C * K * K;
    const int threads = 256;
    // The kernel gets sample n's first float: X + n * C * H * W.
    unrollKernel<<<(total + threads - 1) / threads, threads>>>(X + (size_t)n * C * H * W, Bt, C, H, W, K);
}
// [/lesson:unroll]

// [lesson:gemm-path]
// Unroll + GEMM, one sample at a time, through the project's GEMM unchanged.
// launchLinear computes Y = X * W^T with W stored (out, in). Give it
//   X = F, read as the M x (C*K*K) matrix it already is (p. 465: no copy),
//   W = Bt, P x (C*K*K): stored (out, in) with out = P, so W^T = B,
// and its result, F * B, is M x P: output map m's pixels in row-major order, which is
// sample n's slice of Y in NCHW (p. 466). One scratch Bt serves every sample: launches
// on the default stream run in the order they were issued, on CUDA and on HIP
// (notes/today/ch02-facts.md A5, B3), so sample n + 1's unroll waits for sample n's GEMM.
void launchConvGemm(const float* X, const float* F, float* Y, float* Bt,
                    int N, int C, int H, int W, int M, int K) {
    int P = (H - K + 1) * (W - K + 1);
    for (int n = 0; n < N; ++n) {
        launchUnroll(X, n, Bt, C, H, W, K);
        launchLinear(F, Bt, Y + (size_t)n * M * P, M, C * K * K, P);
    }
}
// [/lesson:gemm-path]
