#define B 8   // tile width; the matrix width is a multiple of B

__global__ void transposeTiles(float* A, int width) {
    __shared__ float tile[B][B];
    int idx = (blockIdx.y * B + threadIdx.y) * width + blockIdx.x * B + threadIdx.x;

    tile[threadIdx.y][threadIdx.x] = A[idx];
    A[idx] = tile[threadIdx.x][threadIdx.y];
}
