__global__ void incorrect_barrier_example(int n) {
    // ...
    if (threadIdx.x % 2 == 0) {
        // ... work of the even threads
        __syncthreads();
    }
    else {
        // ... work of the odd threads
        __syncthreads();
    }
}
