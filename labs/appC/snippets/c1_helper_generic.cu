#define TILE 256

// A plain pointer: the signature names no memory. Under generic addressing
// the address it is given decides whether these loads read shared or global memory.
__device__ float rowMax(const float* p, int n) {
    float m = p[0];
    for (int i = 1; i < n; ++i) m = p[i] > m ? p[i] : m;
    return m;
}

// One block per row of x (row-major, n columns, 1 <= n <= TILE). Thread 0 takes
// the row's maximum twice: from global memory, and from the block's shared copy.
__global__ void rowMaxTwice(const float* x, float* fromGlobal, float* fromShared, int n) {
    __shared__ float tile[TILE];
    const float* row = x + (size_t)blockIdx.x * n;
    for (int i = threadIdx.x; i < n; i += blockDim.x) tile[i] = row[i];
    __syncthreads();
    if (threadIdx.x == 0) {
        fromGlobal[blockIdx.x] = rowMax(row, n);    // an address in global memory
        fromShared[blockIdx.x] = rowMax(tile, n);   // an address in the shared window
    }
}

void launchRowMaxTwice(const float* x, float* fromGlobal, float* fromShared, int rows, int n) {
    rowMaxTwice<<<rows, TILE>>>(x, fromGlobal, fromShared, n);
}
