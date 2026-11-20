// Every thread of every block runs this function. The four built-ins are all
// that tells one thread from another.
__global__ void numberThreads(int* out) {
    // The launch shape: the same values in every thread of the grid.
    int threadsPerBlock = blockDim.x * blockDim.y * blockDim.z;
    // Which block: the same value in all threads of one block.
    int block = blockIdx.x + blockIdx.y * gridDim.x + blockIdx.z * gridDim.x * gridDim.y;
    // Which thread of that block: a different value in each of its threads.
    int thread = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
    out[block * threadsPerBlock + thread] = thread;
}

// The launch of the book's Fig. 3.1: a 2D grid of 3D blocks.
void launchFig31(int* out_d) {
    dim3 dimGrid(2, 2, 1);        // 2 x 2 x 1 blocks
    dim3 dimBlock(4, 2, 2);       // 4 x 2 x 2 threads in each block
    numberThreads<<<dimGrid, dimBlock>>>(out_d);
}
