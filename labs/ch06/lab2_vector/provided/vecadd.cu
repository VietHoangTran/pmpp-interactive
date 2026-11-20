// Provided baseline for lab 6.2: one element per thread, one 4-byte load per input and
// one 4-byte store. This is the vector addition of book Fig. 2.10, the starting point of
// section 6.3 (p. 138). Your two kernels are timed against this one. Do not edit.

__global__ void vecAddKernel(const float* x, const float* y, float* z, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < n) z[i] = x[i] + y[i];
}

void launchVecAdd(const float* x, const float* y, float* z, int n) {
    vecAddKernel<<<(n + 255) / 256, 256>>>(x, y, z, n);
}
