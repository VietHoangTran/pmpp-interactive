// Lab 5.5 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Tiled matrix multiplication whose tile width is chosen at RUN time, using
// dynamically sized shared memory. Book: section 5.6, pp. 116-118.
//
// The book's Fig. 5.14 has several mistakes as printed (two types in the
// declaration, a misspelled parameter, a byte count added to a float pointer).
// This file is the corrected, tested version.

__global__ void tiledMatmulDynamicKernel(const float* M, const float* N, float* P,
                                         int Width, int tileWidth) {
    // One unsized array. Its size in BYTES is the third launch parameter.
    extern __shared__ float tiles[];
    // Carve it into the two tiles by hand. Pointer arithmetic on a float* counts
    // ELEMENTS, so the offset is tileWidth*tileWidth, not that many bytes.
    float* Mds = tiles;
    float* Nds = tiles + tileWidth * tileWidth;

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * tileWidth + ty;
    int Col = blockIdx.x * tileWidth + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < (Width + tileWidth - 1) / tileWidth; ++ph) {
        int mCol = ph * tileWidth + tx;
        int nRow = ph * tileWidth + ty;
        // The tiles are one-dimensional now, so index them the way M and N are indexed.
        Mds[ty * tileWidth + tx] = (Row < Width && mCol < Width) ? M[Row * Width + mCol] : 0.0f;
        Nds[ty * tileWidth + tx] = (nRow < Width && Col < Width) ? N[nRow * Width + Col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < tileWidth; ++k) {
            Pvalue += Mds[ty * tileWidth + k] * Nds[k * tileWidth + tx];
        }
        __syncthreads();
    }
    if (Row < Width && Col < Width) {
        P[Row * Width + Col] = Pvalue;
    }
}

void launchTiledMatmulDynamic(float* M, float* N, float* P, int Width, int tileWidth) {
    dim3 dimBlock(tileWidth, tileWidth);
    unsigned blocks = (Width + tileWidth - 1) / tileWidth;
    dim3 dimGrid(blocks, blocks);
    size_t sharedBytes = 2 * (size_t)tileWidth * tileWidth * sizeof(float);   // both tiles
    tiledMatmulDynamicKernel<<<dimGrid, dimBlock, sharedBytes>>>(M, N, P, Width, tileWidth);
}
