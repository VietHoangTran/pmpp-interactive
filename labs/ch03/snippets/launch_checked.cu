__global__ void fillOnes(float* out, int n) {
    int i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    if (i < n) {
        out[i] = 1.0f;
    }
}

// Launches one block of the given shape. The launch statement has no value to
// test, so the function asks the runtime afterwards.
cudaError_t launchOneBlock(float* out_d, int n, dim3 dimBlock) {
    fillOnes<<<1, dimBlock>>>(out_d, n);
    return cudaGetLastError();    // this host thread's error state; the call also resets it
}

bool blockShapeIsRejected(float* out_d, int n) {
    // 32 x 32 x 2 = 2048 threads in one block. This line compiles.
    cudaError_t err = launchOneBlock(out_d, n, dim3(32, 32, 2));
    return err == cudaErrorInvalidConfiguration;
}
