// Lab 5.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Tiled matrix multiplication with shared memory. Book: section 5.4, Fig. 5.9, p. 108.
//
// Contract: square Width x Width matrices in row-major order, Width a multiple
// of TILE_WIDTH. Lab 5.3 removes both restrictions.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void tiledMatmulKernel(const float* M, const float* N, float* P, int Width) {
    // One copy of each array per block, visible to all of its threads.
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;   // the P element this thread owns
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0;                          // lives in a register
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // Each thread fetches ONE element of M and ONE of N for the whole block.
        Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();   // nobody reads a tile until every thread has written its slot

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();   // nobody overwrites a tile until every thread has finished reading it
    }
    P[Row * Width + Col] = Pvalue;
}

void launchTiledMatmul(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);                 // one thread per element of the output tile
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);  // one block per output tile
    tiledMatmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
