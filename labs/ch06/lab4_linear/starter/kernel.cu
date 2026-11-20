// Lab 6.4 starter (CUDA; also built for HIP and for the CPU emulator).
// The project's linear layer in the framework's weight layout:
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major,
// which is how PyTorch's nn.Linear and llama.cpp's ggml store a weight.
// Book: corner turning, section 6.1 and Fig. 6.4 (pp. 129-130); padding,
// section 6.4 (pp. 139-141); Exercise 2 (p. 155).
//
// This is your lab 5.3 kernel with the second operand replaced by W^T. Element
// (k, o) of W^T is W[o * in + k]. The X tile and the launch are done; the W tile
// is yours:
//   - If thread (ty, tx) fetched element (ty, tx) of the W^T tile, as in lab 5.3,
//     it would read W[Col * in + k]: neighbouring threadIdx.x `in` floats apart,
//     the book's Fig. 6.3. The provided baseline does exactly that.
//   - Fetch the same tile so that neighbouring threadIdx.x read neighbouring
//     floats of one row of W (Fig. 6.4b). Then decide where in the shared tile
//     each value belongs, so that the product loop finds W(Col, k) where it looks.
//   - Look at the shared-memory store you wrote: which way does a warp walk the
//     tile? Section 6.4 says what that does to the banks, and how to declare the
//     tile so it does not.
// Every load must be bounds-checked and zero-filled, as in lab 5.3.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void linearKernel(const float* X, const float* W, float* Y, int batch, int in, int out) {
    __shared__ float Xs[TILE_WIDTH][TILE_WIDTH];
    // TODO 1: declare the shared tile for W.

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;   // row of X and Y (a batch element): valid if < batch
    int Col = blockIdx.x * TILE_WIDTH + tx;   // column of Y (an output feature):   valid if < out

    float Yvalue = 0;
    for (int ph = 0; ph < (in + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        // X tile, as in lab 5.3: neighbouring tx read neighbouring k. Coalesced.
        int xk = ph * TILE_WIDTH + tx;
        Xs[ty][tx] = (Row < batch && xk < in) ? X[Row * in + xk] : 0.0f;

        // TODO 2: load this phase's W tile, one element per thread, corner-turned.
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            // TODO 3: add X(Row, ph*T + i) * W(Col, ph*T + i), both read from shared memory.
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
