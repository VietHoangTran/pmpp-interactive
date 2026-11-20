__global__ void barrier_per_thread(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    s[threadIdx.x] = data[i];                      // phase 0: write my slot
    if (threadIdx.x > 0) {
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // phase 1: read a neighbour's slot
}
