// y = a*x + y for n floats, on the device. It has one defect, on purpose.
// No error checks, to keep the listing short, as in the book's figures.
__global__ void saxpyKernel(float a, const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

void saxpy(float a, const float* x_h, float* y_h, int n) {
    float *x_d, *y_d;
    size_t size = (size_t)n * sizeof(float);
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&y_d, size);
    cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice);

    saxpyKernel<<<(n + 255) / 256, 256>>>(a, x_d, y_d, n);

    cudaMemcpy(y_h, y_d, size, cudaMemcpyDeviceToHost);
    cudaFree(x_d);
    cudaFree(y_d);
}
