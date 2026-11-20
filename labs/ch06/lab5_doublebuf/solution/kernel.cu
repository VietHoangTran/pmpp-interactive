// Lab 6.5 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Double-buffered tiled matrix multiplication: one barrier per phase.
// Book: section 6.7, pp. 146-147 (double buffering), applied to the tiled kernel of
// section 5.4, Fig. 5.9 (lab 5.2).
//
// Contract: square Width x Width matrices in row-major order, Width a multiple
// of TILE_WIDTH (as lab 5.2).
//
// Where this differs from the page:
// - The book's loop (p. 147) swaps two buffer pointers at the end of each
//   iteration. Here the phase number picks the buffer (ph & 1): the same
//   alternation, with the tiles kept as plain shared arrays.
// - Lab 5.2's loop writes first and reads second, so the barrier that guards the
//   false (write-after-read) dependence is its SECOND one, the one after the dot
//   product. That is the one removed here. In the book's toy loop on p. 146 the
//   read comes first, so there it is the first barrier (one sentence on p. 147
//   says "second"; the code and the sentence after it mean the first).
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void doubleBufferedMatmulKernel(const float* M, const float* N, float* P, int Width) {
    // Two tiles per operand: phase ph loads into pair ph & 1 (0, 1, 0, 1, ...).
    __shared__ float Mds[2][TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[2][TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        int buf = ph & 1;
        Mds[buf][ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[buf][ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();   // read-after-write (true dependence): nobody reads the pair until all of it is written

        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[buf][ty][k] * Nds[buf][k][tx];
        }
        // No barrier here. A thread that runs ahead loads phase ph + 1 into the
        // OTHER pair, so it cannot overwrite what a neighbour is still reading.
        // It cannot get two phases ahead either: before it writes this pair again
        // (phase ph + 2) it must pass the barrier of phase ph + 1, and no thread
        // passes that until every thread has finished this dot product.
    }
    P[Row * Width + Col] = Pvalue;
}

void launchDoubleBufferedMatmul(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);                 // one thread per element of the output tile
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);  // one block per output tile
    doubleBufferedMatmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
