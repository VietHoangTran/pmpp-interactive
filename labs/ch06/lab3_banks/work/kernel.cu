// Lab 6.3 starter (CUDA; also built for HIP and for the CPU emulator).
// Matrix transpose through a shared-memory tile. Book: section 6.4 (pp. 139-141);
// section 6.1 (pp. 129-130, Fig. 6.4) for turning a tile in shared memory.
//
// in is rows x cols and out is cols x rows, both row-major: out[c * rows + r] = in[r * cols + c].
// Any rows and cols, not only multiples of the tile. The build defines TILE_WIDTH (16 by
// default; 32 is also built for the benchmark), so do not hard-code 16.
//
// pitch is the distance in words between two rows of the shared tile: element (r, c) of
// the tile is tile[r * pitch + c]. The driver calls your kernel with pitch = TILE_WIDTH
// and with pitch = TILE_WIDTH + 1, so one kernel serves both layouts.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void tiledTransposeKernel(const float* in, float* out, int rows, int cols, int pitch) {
    // TILE_WIDTH rows of TILE_WIDTH + 1 words, enough for the larger pitch. Index it as
    // tile[row * pitch + col]. (Written without parentheses because the CPU emulator's
    // translator does not accept them in a __shared__ array bound.)
    __shared__ float tile[TILE_WIDTH * TILE_WIDTH + TILE_WIDTH];

    // TODO 1: copy this block's input tile into the shared tile. Neighbouring threads
    //         (consecutive threadIdx.x) must read neighbouring addresses of `in`.
    //         Skip the elements that lie outside the matrix.

    // TODO 2: make sure the whole tile is in shared memory before any thread reads it.

    // TODO 3: write this block's output tile. Again neighbouring threads must write
    //         neighbouring addresses of `out`, so each thread takes its value from the
    //         tile down a column, not along a row. Skip elements outside the output.
}

void launchTiledTranspose(const float* in, float* out, int rows, int cols, int pitch) {
    // TODO 4: one TILE_WIDTH x TILE_WIDTH block per tile of the input, enough blocks to
    //         cover a matrix that is not a multiple of the tile in either direction.
}
