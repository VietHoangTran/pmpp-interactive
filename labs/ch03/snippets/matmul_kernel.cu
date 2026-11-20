__global__ void MatrixMulKernel(float* M, float* N,
                                float* P, int Width) {
    int row = blockIdx.y*blockDim.y+threadIdx.y;
    int col = blockIdx.x*blockDim.x+threadIdx.x;
    if ((row < Width) && (col < Width)) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row*Width+k]*N[k*Width+col];
        }
        P[row*Width+col] = Pvalue;
    }
}

#define BLOCK_WIDTH 16

// Not printed in the book: a square grid of square blocks, rounded up to whole blocks.
void launchMatrixMul(float* M_d, float* N_d, float* P_d, int Width) {
    int blocks = (Width + BLOCK_WIDTH - 1) / BLOCK_WIDTH;   // ceil(Width / BLOCK_WIDTH)
    dim3 dimGrid(blocks, blocks, 1);                        // x covers the columns of P, y its rows
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(M_d, N_d, P_d, Width);
}
