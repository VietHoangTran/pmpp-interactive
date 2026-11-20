// Chapter 2 in two functions: one thread adds one pair of elements,
// and the host launches enough 256-thread blocks to cover all n of them.
__global__ void vecAddKernel(float* A, float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void launchVecAdd(float* A_d, float* B_d, float* C_d, int n) {
    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    vecAddKernel<<<blocks, threads>>>(A_d, B_d, C_d, n);
}
