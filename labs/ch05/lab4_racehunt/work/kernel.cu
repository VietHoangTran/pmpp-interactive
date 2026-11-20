// Lab 5.4 starter: a tiled, boundary-checked matrix multiplication for square
// Width x Width matrices of any width (after book Fig. 5.13, p. 115).
//
// This kernel has THREE independent defects. It compiles, and it can pass when
// Width is a multiple of TILE_WIDTH. Find and fix all three.
// Run it on the CPU emulator first: its thread schedules are deterministic, so
// a synchronisation defect fails every time there instead of occasionally.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void tiledMatmulKernel(const float* M, const float* N, float* P, int Width) {
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < (Width + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        if (Row < Width && ph * TILE_WIDTH + tx < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];

        if (ph * TILE_WIDTH + tx < Width && Col < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
    }
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
}

void launchTiledMatmul(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    unsigned blocks = (Width + TILE_WIDTH - 1) / TILE_WIDTH;
    dim3 dimGrid(blocks, blocks);
    tiledMatmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
