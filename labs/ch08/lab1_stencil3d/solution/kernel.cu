// Lab 8.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// A 3D 7-point stencil sweep. Book: sections 8.2 to 8.6, Figs. 8.6 (p. 188) and
// 8.12 (p. 197).
//
// Contract: an N x N x N float grid, row-major with x fastest (i = z, j = y, k = x).
// Only interior points are computed; the boundary layer of `out` is left alone (the
// driver has already copied `in` into `out`). Weights: see ../stencil.h.
//
// The build defines TILE_WIDTH (32 by default; 16 and 8 are also built for the
// benchmark). The register-tiled kernel's blocks are TILE_WIDTH x TILE_WIDTH threads.
#include "../stencil.h"

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM (IN_TILE_DIM - 2)

// ---------------------------------------------------------------- Part 1: Fig. 8.6
// One thread per grid point, every operand read from global memory.
// Differs from the page: `in` is const and the weights arrive as one argument.
__global__ void stencilBasicKernel(const float* in, float* out, unsigned int N, StencilCoeffs c) {
    unsigned int i = blockIdx.z * blockDim.z + threadIdx.z;
    unsigned int j = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
        out[i * N * N + j * N + k] = c.c0 * in[i * N * N + j * N + k]
                                   + c.c1 * in[i * N * N + j * N + (k - 1)]
                                   + c.c2 * in[i * N * N + j * N + (k + 1)]
                                   + c.c3 * in[i * N * N + (j - 1) * N + k]
                                   + c.c4 * in[i * N * N + (j + 1) * N + k]
                                   + c.c5 * in[(i - 1) * N * N + j * N + k]
                                   + c.c6 * in[(i + 1) * N * N + j * N + k];
    }
}

void launchStencilBasic(const float* in, float* out, unsigned int N, StencilCoeffs c) {
    dim3 dimBlock(8, 8, 8);                          // exercise 1b's block
    unsigned int blocks = (N + 7) / 8;               // threads cover every point, boundary included
    dim3 dimGrid(blocks, blocks, blocks);
    stencilBasicKernel<<<dimGrid, dimBlock>>>(in, out, N, c);
}

// ---------------------------------------------------------------- Part 2: Fig. 8.12
// A 2D block marches through `depth` output planes in z. Each thread keeps its own
// column's previous, current and next values in registers; only the current plane
// is mirrored in shared memory, because only it is read by other threads (the four
// x-y neighbours).
//
// Differs from the page:
//   - The slab depth is the run-time argument `depth`; the book's is OUT_TILE_DIM
//     (`iStart = blockIdx.z*OUT_TILE_DIM`, loop to `iStart + OUT_TILE_DIM`).
//   - c3 multiplies the y-1 neighbour and c4 the y+1 neighbour, as in Figs. 8.6 and
//     8.8; Fig. 8.12 prints them the other way round.
//   - The last line of the loop stores the register inNext into inCurr_s; the page
//     prints `= inNext_s`, a variable this kernel does not have.
//   - The three registers start at 0 so that a thread outside the grid, which loads
//     nothing, never copies an undefined value. The page leaves them uninitialised.
__global__ void stencilRegTiledKernel(const float* in, float* out, unsigned int N, unsigned int depth,
                                      StencilCoeffs c) {
    int iStart = blockIdx.z * depth;
    int j = blockIdx.y * OUT_TILE_DIM + threadIdx.y - 1;
    int k = blockIdx.x * OUT_TILE_DIM + threadIdx.x - 1;
    float inPrev = 0.0f;
    __shared__ float inCurr_s[IN_TILE_DIM][IN_TILE_DIM];
    float inCurr = 0.0f;
    float inNext = 0.0f;
    if (iStart - 1 >= 0 && iStart - 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
        inPrev = in[(iStart - 1) * N * N + j * N + k];
    }
    if (iStart >= 0 && iStart < N && j >= 0 && j < N && k >= 0 && k < N) {
        inCurr = in[iStart * N * N + j * N + k];
        inCurr_s[threadIdx.y][threadIdx.x] = inCurr;
    }
    // Every thread runs every iteration and reaches both barriers, including threads
    // that compute nothing: they still load their element of the halo.
    for (int i = iStart; i < iStart + depth; ++i) {
        if (i + 1 >= 0 && i + 1 < N && j >= 0 && j < N && k >= 0 && k < N) {
            inNext = in[(i + 1) * N * N + j * N + k];
        }
        __syncthreads();   // inCurr_s holds the whole current plane
        if (i >= 1 && i < N - 1 && j >= 1 && j < N - 1 && k >= 1 && k < N - 1) {
            if (threadIdx.y >= 1 && threadIdx.y < IN_TILE_DIM - 1 &&
                threadIdx.x >= 1 && threadIdx.x < IN_TILE_DIM - 1) {
                out[i * N * N + j * N + k] = c.c0 * inCurr
                                           + c.c1 * inCurr_s[threadIdx.y][threadIdx.x - 1]
                                           + c.c2 * inCurr_s[threadIdx.y][threadIdx.x + 1]
                                           + c.c3 * inCurr_s[threadIdx.y - 1][threadIdx.x]
                                           + c.c4 * inCurr_s[threadIdx.y + 1][threadIdx.x]
                                           + c.c5 * inPrev
                                           + c.c6 * inNext;
            }
        }
        __syncthreads();   // nobody overwrites inCurr_s until every neighbour has read it
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[threadIdx.y][threadIdx.x] = inNext;
    }
}

void launchStencilRegTiled(const float* in, float* out, unsigned int N, unsigned int depth, StencilCoeffs c) {
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM, 1);                 // one x-y plane of the input tile
    unsigned int tiles = (N + OUT_TILE_DIM - 1) / OUT_TILE_DIM;
    dim3 dimGrid(tiles, tiles, (N + depth - 1) / depth);         // z: one block per slab of `depth` planes
    stencilRegTiledKernel<<<dimGrid, dimBlock>>>(in, out, N, depth, c);
}
