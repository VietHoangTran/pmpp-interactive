__global__ void barrier_guarded(float* data, int n) {
    __shared__ float s[4];                         // the block has 4 threads
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    bool mine = i < n;                             // threads past the end stay, and do nothing
    if (mine) s[threadIdx.x] = data[i];            // phase 0: write my slot
    __syncthreads();
    if (mine) data[i] = s[0];                      // phase 1: read thread 0's slot
}
