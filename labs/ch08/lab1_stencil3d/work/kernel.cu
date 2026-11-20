// Lab 8.1 starter (CUDA; also built for HIP and for the CPU emulator).
// A 3D 7-point stencil sweep. Book: sections 8.2 to 8.6, pp. 187-198.
//
// Contract: an N x N x N float grid, row-major with x fastest: point (i, j, k), with
// i the z index, j the y index and k the x index, is element i*N*N + j*N + k.
// Compute only the interior points (1 <= i, j, k <= N-2). Leave the boundary layer of
// `out` alone: the driver has already copied `in` into `out`. The seven weights and
// the neighbour each one multiplies are in ../stencil.h.
//
// The build defines TILE_WIDTH (32 by default; 16 and 8 are also built for the
// benchmark), so do not hard-code 32 anywhere.
#include "../stencil.h"

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// ---------------------------------------------------------------- Part 1: Fig. 8.6
// One thread per grid point, every operand read from global memory.
__global__ void stencilBasicKernel(const float* in, float* out, unsigned int N, StencilCoeffs c) {
    // TODO 1: which grid point (i, j, k) does this thread own? (3D blocks, 3D grid)

    // TODO 2: if it is an interior point, write the weighted sum of the point and its
    //         six neighbours to out. Neighbours in x are 1 element apart; how far
    //         apart are neighbours in y, and in z?
}

void launchStencilBasic(const float* in, float* out, unsigned int N, StencilCoeffs c) {
    // TODO 3: 8 x 8 x 8 blocks (exercise 1b), enough of them to give every grid
    //         point, boundary included, its own thread.
}

// ---------------------------------------------------------------- Part 2: Fig. 8.12
// A block of IN_TILE_DIM x IN_TILE_DIM threads, one x-y plane of the input tile,
// marches through `depth` output planes in z, starting at plane blockIdx.z * depth.
// In x and y the block's input tile starts one point before its output tile.
//
// Which values does only this thread ever read? Keep those in registers. Which
// values do other threads read too? Only those need shared memory.
__global__ void stencilRegTiledKernel(const float* in, float* out, unsigned int N, unsigned int depth,
                                      StencilCoeffs c) {
    // TODO 4: the first plane of this block's slab, and this thread's j and k.

    // TODO 5: declare the on-chip storage for the three planes the stencil needs at
    //         each step (previous, current, next). One of them is read by other
    //         threads; the others are not.

    // TODO 6: before the loop, load this thread's values of the plane before the
    //         slab and of the slab's first plane, each only if it lies inside the grid.

    // TODO 7: the loop, one output plane per iteration:
    //   a. load this thread's value of the next plane (only if it lies inside the grid)
    //   b. wait until the whole block has its current plane in shared memory
    //   c. if this thread's point is inside the grid's interior AND the thread is not
    //      on the edge of the block, write the output from the seven values
    //   d. wait until every thread has finished reading the current plane
    //   e. move the window one plane on: previous <- current <- next, and keep the
    //      shared copy of the current plane up to date
    //
    // Every thread of the block must reach both barriers on every iteration, so do not
    // return early: a thread that computes no output still loads part of the halo.
}

void launchStencilRegTiled(const float* in, float* out, unsigned int N, unsigned int depth, StencilCoeffs c) {
    // TODO 8: blocks of IN_TILE_DIM x IN_TILE_DIM threads. In x and y, one block per
    //         OUT_TILE_DIM-wide output tile; in z, one block per slab of `depth` planes.
}
