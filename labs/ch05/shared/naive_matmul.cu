// Provided baseline: one thread per output element, every operand fetched from
// global memory (the kernel of book Fig. 3.11, p. 61). Labs 2 and 3 time their
// kernels against this one.
#ifndef PMPP_NAIVE_BLOCK
#define PMPP_NAIVE_BLOCK 16
#endif

__global__ void naiveMatmulKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}

void launchNaiveMatmul(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(PMPP_NAIVE_BLOCK, PMPP_NAIVE_BLOCK);
    unsigned blocks = (Width + PMPP_NAIVE_BLOCK - 1) / PMPP_NAIVE_BLOCK;
    dim3 dimGrid(blocks, blocks);
    naiveMatmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
