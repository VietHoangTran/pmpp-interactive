// Lab 6.4 reference solution (CUDA; also built for HIP and for the CPU emulator).
// The project's linear layer in the framework's weight layout, corner-turned:
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major,
// which is how PyTorch's nn.Linear and llama.cpp's ggml store a weight.
// Book: corner turning, section 6.1 and Fig. 6.4 (pp. 129-130); padding the
// turned tile, section 6.4 (pp. 139-141); Exercise 2 (p. 155). The book gives no
// listing for Exercise 2: this is our version of it, in the layout the project
// needs. W read as W^T is exactly the book's column-major N (p. 128).
// Project milestone M2 "measured and tuned". Chapter 15 builds on this signature.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void linearKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    __shared__ float Xs[TILE_WIDTH][TILE_WIDTH];
    // Ws[i][j] holds W(o = bx*T + j, k = ph*T + i): the tile of W^T this block needs.
    // One extra column: the corner-turned store below writes down a column of Ws,
    // and without it those writes would fall in the same few banks (section 6.4).
    __shared__ float Ws[TILE_WIDTH][TILE_WIDTH + 1];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;   // row of X and Y (a batch element): valid if < batch
    int Col = blockIdx.x * TILE_WIDTH + tx;   // column of Y (an output feature):   valid if < out

    float Yvalue = 0;
    for (int ph = 0; ph < (in + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        // X tile, as in lab 5.3: neighbouring tx read neighbouring k. Coalesced.
        int xk = ph * TILE_WIDTH + tx;
        Xs[ty][tx] = (Row < batch && xk < in) ? X[Row * in + xk] : 0.0f;

        // W tile, corner-turned (Fig. 6.4b): tx and ty swap roles for this load, so
        // neighbouring tx read neighbouring k along ONE row of W. Coalesced.
        int wo = blockIdx.x * TILE_WIDTH + ty;   // the row of W (output feature) this thread fetches
        int wk = ph * TILE_WIDTH + tx;           // the column of W (input) it fetches
        Ws[tx][ty] = (wo < out && wk < in) ? W[wo * in + wk] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Yvalue += Xs[ty][i] * Ws[i][tx];   // Ws read along a row: no conflict
        }
        __syncthreads();
    }
    if (Row < batch && Col < out) {
        Y[Row * out + Col] = Yvalue;
    }
}

void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((out + TILE_WIDTH - 1) / TILE_WIDTH,      // x covers the outputs (columns of Y)
                 (batch + TILE_WIDTH - 1) / TILE_WIDTH);   // y covers the batch (rows of Y)
    linearKernel<<<dimGrid, dimBlock>>>(X, W, Y, batch, in, out);
}
