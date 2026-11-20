__global__ void barrier_loop(float* data) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // odd threads go round the loop twice, even threads once
    for (unsigned k = 0; k <= threadIdx.x % 2; ++k) {
        s[threadIdx.x] = data[i] + k;              // one round: write my slot
        __syncthreads();
    }
    data[i] = s[(threadIdx.x + 1) % blockDim.x];   // after the loop: read a neighbour's slot
}
