__global__ void vecAddKernel(float* A, float* B, float* C, int n);   // as in chapter 2

// Both shapes written out: 32 blocks of 128 threads.
void launchWritten(float* A_d, float* B_d, float* C_d, int n) {
    dim3 dimGrid(32, 1, 1);       // the grid, in blocks:  x, y, z
    dim3 dimBlock(128, 1, 1);     // a block, in threads:  x, y, z
    vecAddKernel<<<dimGrid, dimBlock>>>(A_d, B_d, C_d, n);
}

// The same launch. Each integer becomes a dim3 whose y and z are 1.
void launchShort(float* A_d, float* B_d, float* C_d, int n) {
    vecAddKernel<<<32, 128>>>(A_d, B_d, C_d, n);
}
