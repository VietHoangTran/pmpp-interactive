// Lab 6.5 starter (CUDA; also built for HIP and for the CPU emulator).
// Book: section 6.7, pp. 146-147 (double buffering), applied to the tiled kernel of
// section 5.4, Fig. 5.9 (lab 5.2).
//
// This is lab 5.2's kernel with its second __syncthreads() deleted. It compiles,
// and it is wrong. Run it on the race backend before you change anything:
//     python3 labs/pmpp.py run 6.5 --backend race
// and find the two lines of this file that ThreadSanitizer names.
//
// Your task: make it correct with ONE __syncthreads() per phase. Do not put the
// second barrier back.
//
// Contract: square Width x Width matrices in row-major order, Width a multiple
// of TILE_WIDTH. The build defines TILE_WIDTH (16 by default; 8 and 32 are also
// built for the benchmark), so do not hard-code 16 anywhere.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void doubleBufferedMatmulKernel(const float* M, const float* N, float* P, int Width) {
    // TODO 1: the deleted barrier enforced one of the two kinds of dependence the
    //         book names in section 6.7. Which one, and what makes it exist at all?
    //         Change these declarations so that it no longer exists.
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;
    int Col = blockIdx.x * TILE_WIDTH + tx;

    float Pvalue = 0;
    for (int ph = 0; ph < Width / TILE_WIDTH; ++ph) {
        // TODO 2: decide where this phase's tiles go, and load them there.
        Mds[ty][tx] = M[Row * Width + (ph * TILE_WIDTH + tx)];
        Nds[ty][tx] = N[(ph * TILE_WIDTH + ty) * Width + Col];
        __syncthreads();   // the one barrier per phase that you keep

        // TODO 3: read the tiles this phase loaded.
        for (int k = 0; k < TILE_WIDTH; ++k) {
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        // Lab 5.2 had a second __syncthreads() here. It stays deleted.
        // TODO 4: once the tests pass, write here why a thread that races ahead
        //         cannot overwrite a tile a neighbour is still reading, not even
        //         two phases later.
    }
    P[Row * Width + Col] = Pvalue;
}

// The launch is lab 5.2's and is correct: double buffering changes the kernel only.
void launchDoubleBufferedMatmul(float* M, float* N, float* P, int Width) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);                 // one thread per element of the output tile
    dim3 dimGrid(Width / TILE_WIDTH, Width / TILE_WIDTH);  // one block per output tile
    doubleBufferedMatmulKernel<<<dimGrid, dimBlock>>>(M, N, P, Width);
}
