// The host stub that section 2.4 steps through call by call: the book's
// Fig. 2.13 with this course's _h and _d names, and, like the book's listing,
// with no error checks. Lab 2.1's solution is this function with every call
// checked. The kernel is opened up in section 2.5.
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}

void vecAdd(const float* A_h, const float* B_h, float* C_h, int n) {
    int size = n * sizeof(float);
    float *A_d, *B_d, *C_d;

    // Part 1: allocate device memory for A, B and C; copy A and B to it
    cudaMalloc((void**)&A_d, size);
    cudaMalloc((void**)&B_d, size);
    cudaMalloc((void**)&C_d, size);
    cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice);
    cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice);

    // Part 2: launch a grid of threads that does the addition
    vecAddKernel<<<(n + 255) / 256, 256>>>(A_d, B_d, C_d, n);

    // Part 3: copy C back to the host; free the three device arrays
    cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost);
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}
