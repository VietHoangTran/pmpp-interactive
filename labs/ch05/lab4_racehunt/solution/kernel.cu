// Lab 5.4 reference solution: the three defects in the starter, fixed.
// Square Width x Width matrices of any width (book Fig. 5.13, p. 115).
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
        // FIX 1 (both loads): a slot with no element behind it must be zeroed. The starter
        // had no else branches, so such slots kept whatever an earlier phase had put there
        // and stale M values were multiplied by stale N values.
        if (Row < Width && ph * TILE_WIDTH + tx < Width)
            Mds[ty][tx] = M[Row * Width + ph * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        // FIX 2: the N element is `ty` rows down its tile, so the row index to check is
        // ph*TILE_WIDTH + ty. The starter checked ph*TILE_WIDTH + tx, copied from the M test.
        if (ph * TILE_WIDTH + ty < Width && Col < Width)
            Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        // FIX 3: without this barrier a thread that finishes its partial sum early starts
        // the next phase and overwrites tile slots its neighbours are still reading
        // (a write-after-read, or false, dependence).
        __syncthreads();
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
