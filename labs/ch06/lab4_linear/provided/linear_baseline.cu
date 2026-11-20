// Provided baseline for lab 6.4 (the "before"): Y[batch x out] = X[batch x in] * W^T,
// W stored (out, in) row-major. It is the lab 5.3 tiled kernel with the second
// operand read as W^T the obvious way: thread (ty, tx) loads element (ty, tx) of
// the W^T tile, which is W[Col * in + k]. Neighbouring threadIdx.x then read
// floats `in` apart, the book's Fig. 6.3 pattern. Correct, but the W loads are
// not coalesced. Built with the same TILE_WIDTH as your kernel.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void linearBaselineKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    __shared__ float Xs[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Ws[TILE_WIDTH][TILE_WIDTH];   // Ws[i][j] = W(o = bx*T + j, k = ph*T + i)

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;   // batch element: valid if < batch
    int Col = blockIdx.x * TILE_WIDTH + tx;   // output feature: valid if < out

    float Yvalue = 0;
    for (int ph = 0; ph < (in + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int xk = ph * TILE_WIDTH + tx;
        int wk = ph * TILE_WIDTH + ty;
        Xs[ty][tx] = (Row < batch && xk < in) ? X[Row * in + xk] : 0.0f;
        Ws[ty][tx] = (Col < out && wk < in) ? W[Col * in + wk] : 0.0f;   // stride `in` across tx
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Yvalue += Xs[ty][i] * Ws[i][tx];
        }
        __syncthreads();
    }
    if (Row < batch && Col < out) {
        Y[Row * out + Col] = Yvalue;
    }
}

void launchLinearBaseline(const float* X, const float* W, float* Y, int batch, int in, int out) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((out + TILE_WIDTH - 1) / TILE_WIDTH, (batch + TILE_WIDTH - 1) / TILE_WIDTH);
    linearBaselineKernel<<<dimGrid, dimBlock>>>(X, W, Y, batch, in, out);
}
