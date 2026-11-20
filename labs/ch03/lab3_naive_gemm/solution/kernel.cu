// Lab 3.3 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Naive GEMM, one thread per element of C:
//   C[j x l] = A[j x k] * B[k x l], row-major float, any sizes.
// Book: section 3.4, Fig. 3.11 (p. 61). First half of project milestone M1.
//
// As a linear layer: j is the batch, k the inputs, l the outputs, and B holds the
// weight as (in, out). The argument order and the storage convention are those of
// lab 5.3's gemmKernel, so the tiled kernel can replace this one later.
//
// Where this file differs from the page:
//   - Fig. 3.11 is square only: one Width stands for all three sizes (p. 61).
//     Here j = rows of A and of C, k = columns of A = rows of B, l = columns of
//     B and of C.
//   - The loop index is i, because k is now a size.
//   - M, N, P are called A, B, C, and the two inputs are const.
//   - The book prints no launch code for this kernel. launchNaiveGemm is ours,
//     consistent with Fig. 3.12 (p. 62), with the grid sized by integer round-up.
#define BLOCK_WIDTH 16

__global__ void naiveGemmKernel(const float* A, const float* B, float* C, int j, int k, int l) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;   // row of C, and the row of A to read
    int col = blockIdx.x*blockDim.x + threadIdx.x;   // column of C, and the column of B to read
    if ((row < j) && (col < l)) {                    // the grid is whole blocks: some threads own nothing
        float Cvalue = 0;
        for (int i = 0; i < k; ++i) {
            // A is walked along a row (stride 1), B down a column (stride l).
            Cvalue += A[row*k + i]*B[i*l + col];
        }
        C[row*l + col] = Cvalue;
    }
}

void launchNaiveGemm(float* A, float* B, float* C, int j, int k, int l) {
    dim3 dimBlock(BLOCK_WIDTH, BLOCK_WIDTH, 1);
    dim3 dimGrid((l + BLOCK_WIDTH - 1)/BLOCK_WIDTH,    // x covers the columns of C
                 (j + BLOCK_WIDTH - 1)/BLOCK_WIDTH,    // y covers the rows of C
                 1);
    naiveGemmKernel<<<dimGrid, dimBlock>>>(A, B, C, j, k, l);
}
