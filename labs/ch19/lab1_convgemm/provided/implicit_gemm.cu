// Lab 19.1, provided (CUDA; also built for HIP and for the CPU emulators). You do not edit
// this file; the benchmark times it beside your two paths, and section 19.3 shows it.
//
// The book's Fig. 19.11 (p. 471): the tiled matrix multiplication of Fig. 5.9 computing
// one sample's Y = F * B, where F is the filter bank read as an M x (C*K*K) matrix and B,
// the (C*K*K) x (H_out*W_out) unrolled input, is never stored. Each thread fetches its
// element of the B tile straight from X with the index map of Eq. 19.5 (p. 470):
//   B[u, v] = X[n, u / K^2, (u % K^2) / K + v / W_out, (u % K^2) % K + v % W_out]
//
// Where this file differs from the page (notes/ch19.md, "Unclear in the PDF"):
//   1. The phase count rounds up: (C*K*K + TILE_WIDTH - 1) / TILE_WIDTH. The page's
//      (C*K*K) / TILE_WIDTH truncates, so with LeNet C1's C*K*K = 25 it runs one phase
//      and drops 9 of the 25 terms of every output.
//   2. Bounds checks, which the book leaves as an exercise (p. 470): the F tile load
//      checks Row < M and its column < C*K*K, the X gather checks u < C*K*K and
//      v < H_out*W_out, and both store 0 when out of range, so the product loop can run
//      the full tile. The store to Y checks Row < M and Col < H_out*W_out.
//   3. The host code, which the book does not show (it states only that grid z is the
//      sample, p. 470): blocks of TILE_WIDTH x TILE_WIDTH threads, grid
//      (ceil(H_out*W_out / T), ceil(M / T), N). x runs along the columns of Y (output
//      pixels), y along its rows (output maps), as in Fig. 5.9.
//   4. const on the inputs; W_out and H_out are named once instead of written W-K+1.
// The arithmetic is the book's: no 64-bit indices, and an integer divide and remainder
// for every element gathered (the "additional indexing arithmetic" of p. 474).
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void ConvLayer_MM_Kernel(int C, int M, int H, int W, int K,
                                    const float* F, const float* X, float* Y) {
    __shared__ float Fds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y, bz = blockIdx.z;   // bz: the sample n
    int tx = threadIdx.x, ty = threadIdx.y;

    int H_out = H - K + 1, W_out = W - K + 1;
    int P = H_out * W_out;     // columns of Y and of B: output pixels
    int CKK = C * K * K;       // columns of F, rows of B

    int Row = by * TILE_WIDTH + ty;   // row of Y: the output map m
    int Col = bx * TILE_WIDTH + tx;   // column of Y: the output pixel, row-major

    float Pvalue = 0;
    for (int ph = 0; ph < (CKK + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {   // (1) ceiling
        // F needs no unrolling: the M x C x K x K array is already an M x (C*K*K) matrix.
        int fc = ph * TILE_WIDTH + tx;
        Fds[ty][tx] = (Row < M && fc < CKK) ? F[Row * CKK + fc] : 0.0f;          // (2)

        // The B element this thread loads, gathered from X with Eq. 19.5.
        int u = ph * TILE_WIDTH + ty;
        int v = Col;
        if (u < CKK && v < P) {                                                  // (2)
            Bds[ty][tx] = X[bz * (C * H * W) + (u / (K * K)) * (H * W)
                            + ((u % (K * K)) / K + v / W_out) * W
                            + ((u % (K * K)) % K + v % W_out)];
        } else {
            Bds[ty][tx] = 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Fds[ty][k] * Bds[k][tx];
        }
        __syncthreads();
    }
    if (Row < M && Col < P) {                                                    // (2)
        Y[bz * M * P + Row * P + Col] = Pvalue;
    }
}

void launchConvImplicitGemm(const float* X, const float* F, float* Y,
                            int N, int C, int H, int W, int M, int K) {                // (3)
    int P = (H - K + 1) * (W - K + 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    dim3 dimGrid((P + TILE_WIDTH - 1) / TILE_WIDTH,   // x: tiles of output pixels (columns of Y)
                 (M + TILE_WIDTH - 1) / TILE_WIDTH,   // y: tiles of output maps (rows of Y)
                 N);                                   // z: the sample
    ConvLayer_MM_Kernel<<<dimGrid, dimBlock>>>(C, M, H, W, K, F, X, Y);
}
