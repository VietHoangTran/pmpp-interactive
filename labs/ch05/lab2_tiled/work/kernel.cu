// Lab 5.2 starter (CUDA; also built for HIP and for the CPU emulator).
// Tiled matrix multiplication with shared memory. Book: section 5.4, pp. 108-112.
//
// Contract: square Width x Width matrices in row-major order, Width a multiple
// of TILE_WIDTH. The build defines TILE_WIDTH (16 by default; 8 and 32 are also
// built for the benchmark), so do not hard-code 16 anywhere.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void tiledMatmulKernel(const float* M, const float* N, float* P, int Width) {
    // TODO 1: declare the two shared-memory tiles (one for M, one for N).

    // TODO 2: which row and column of P does this thread own?
    //         (Blocks are TILE_WIDTH x TILE_WIDTH threads.)

    // TODO 3: loop over the phases. In each phase:
    //   a. this thread loads ONE element of M and ONE element of N into the tiles
    //   b. wait until the whole block has loaded
    //   c. add this phase's TILE_WIDTH products to a running sum held in a register
    //   d. wait until the whole block has finished reading the tiles

    // TODO 4: store the result to P.
}

void launchTiledMatmul(float* M, float* N, float* P, int Width) {
    // TODO 5: one block per output tile, one thread per element of the tile.
}
