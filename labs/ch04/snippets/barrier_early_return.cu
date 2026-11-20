__global__ void barrier_early_return(float* data, int n) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;                            // threads past the end of the data leave here
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    __syncthreads();
    data[i] = s[0];                                // phase 1: read thread 0's slot
}
