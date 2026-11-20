// Lab 5.1 starter (CUDA; also built for HIP and for the CPU emulator).
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
// One thread computes one element of P = M * N for square Width x Width
// matrices stored row-major (book Fig. 3.11, p. 61). Write it from memory if
// you can: you will analyse and then rewrite this kernel for the rest of the
// chapter.
__global__ void matmulKernel(const float* M, const float* N, float* P, int Width) {
    // TODO 1: which row and column of P does this thread own?
    // TODO 2: guard against threads that fall outside the matrix.
    // TODO 3: accumulate the dot product of that row of M and that column of N
    //         in a local variable, then store it to P once.
}

void launchMatmul(float* M, float* N, float* P, int Width) {
    // TODO 4: choose a 2D block size and a grid that covers a Width x Width
    //         output for ANY Width (the tests use widths such as 5, 33 and 257),
    //         then launch matmulKernel.
}
