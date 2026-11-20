// Divide every row of a rows x cols matrix by that row's first element, in place.
__global__ void scaleRowsKernel(float* X, int rows, int cols) {
    int row = blockIdx.y*blockDim.y + threadIdx.y;
    int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < rows && col < cols) {
        X[row*cols + col] = X[row*cols + col] / X[row*cols];
    }
}

void launchScaleRows(float* X, int rows, int cols) {
    dim3 dimBlock(16, 16, 1);
    dim3 dimGrid((cols + 15)/16, (rows + 15)/16, 1);
    scaleRowsKernel<<<dimGrid, dimBlock>>>(X, rows, cols);
}
