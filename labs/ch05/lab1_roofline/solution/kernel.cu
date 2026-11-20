// Lab 5.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Book: section 5.1 (pp. 93-97) and the kernel of Fig. 3.11 (p. 61).

// ---------- provided probe 1: memory bandwidth ----------
// Vector addition touches 12 bytes of global memory per floating-point add
// (0.083 FLOP/B), so its speed is set almost entirely by memory bandwidth.
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) C[i] = A[i] + B[i];
}

void launchVecAdd(float* A, float* B, float* C, int n) {
    vecAddKernel<<<(n + 255) / 256, 256>>>(A, B, C, n);
}

// ---------- provided probe 2: arithmetic throughput ----------
// Each thread loads one value, does `iters` multiply-adds entirely in
// registers, and stores one value: almost no memory traffic per operation.
__global__ void fmaChainKernel(const float* in, float* out, int n, int iters, float a, float b) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float x = in[i];
        for (int it = 0; it < iters; ++it) x = x * a + b;
        out[i] = x;
    }
}

void launchFmaChain(float* in, float* out, int n, int iters, float a, float b) {
    fmaChainKernel<<<(n + 255) / 256, 256>>>(in, out, n, iters, a, b);
}

// ---------- your part: the naive matrix multiplication ----------
__global__ void matmulKernel(const float* M, const float* N, float* P, int Width) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < Width && col < Width) {          // the grid may overhang the matrix
        float Pvalue = 0;
        for (int k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];   // row of M, column of N
        }
        P[row * Width + col] = Pvalue;
    }
}

void launchMatmul(float* M, float* N, float* P, int Width) {
    const int B = 16;
    dim3 dimBlock(B, B);
    dim3 dimGrid((Width + B - 1) / B, (Width + B - 1) / B);   // round up so every element is covered
    matmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
