// Provided baseline: lab 5.2's tiled matrix multiplication, two barriers per phase
// (book section 5.4, Fig. 5.9, p. 108). Lab 6.5 times your double-buffered kernel
// against this one. Renamed so that both can live in one program; otherwise the
// same as labs/ch05/lab2_tiled/solution/kernel.cu.
//
// Contract: square Width x Width matrices in row-major order, Width a multiple
// of TILE_WIDTH.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void twoBarrierMatmulKernel(const float* M, const float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();   // read-after-write: nobody reads a tile until every slot is written

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();   // write-after-read: nobody overwrites a tile until everyone has read it
    }
    P[Row * Width + Col] = Pvalue;
}

void launchTwoBarrierMatmul(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);
    twoBarrierMatmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
