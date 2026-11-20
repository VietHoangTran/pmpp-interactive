// Lab 16.1 starter (CUDA; also built for HIP and for the CPU emulator).
// The Smith-Waterman scoring table, tiled two ways.
// Book: section 16.6 (pp. 384-389, Figs. 16.8 to 16.12), square tiles;
//       section 16.7 (pp. 389-396, Figs. 16.15 to 16.20), sheared tiles ("hypertiles").
//
// Contract: H is (L_seq + 1) x (L_seq + 1) ints, row-major, with row 0 and column 0
// already zero; never write them. a indexes rows, b columns. L = L_seq + 1 is the
// width of a table row. A tile is W x W cells (W = threads per block); T =
// ceil(L_seq / W) tiles per side. In shared memory, tile element (row, col) is
// swTile[row * stride + col], stride = W + pad: pad = 1 leaves one unused int after
// every tile row.
//
// Write all index arithmetic in int. blockIdx.x, threadIdx.x and blockDim.x are
// unsigned: convert them before you subtract. (The book's `d - blockIdx.x` relies on
// how an out-of-range unsigned value converts to int, which C++17 leaves to the
// compiler.)
#include <algorithm>
#include <cstddef>

#include "../sw.h"

// Fig. 16.11: the largest of four ints.
__device__ inline int max4(int a, int b, int c, int d) {
    // TODO
    return 0;
}

// ---------------------------------------------------------------- square tiles
// Fig. 16.10, with the row stride. Each returns one neighbour (north, west or
// north-west) of table cell (r, q), which is tile cell (r_tile, q_tile). A neighbour
// inside this tile is in swTile; one outside it was stored in H by an earlier launch.
__device__ inline int sqLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    // TODO
    return 0;
}
__device__ inline int sqLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    // TODO
    return 0;
}
__device__ inline int sqLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    // TODO
    return 0;
}

// Fig. 16.12: copy the tile to H. Here the thread index is the tile COLUMN, so the
// block writes each tile row to H as one contiguous piece. Skip cells outside the table.
__device__ inline void sqStoreTile(int* H, const int* swTile, int L, int W, int stride, int tile_row, int tile_col) {
    // TODO
}

// Fig. 16.9. One block per tile of front d; this block's tile row is firstRow +
// blockIdx.x. Inside the kernel the thread index is the tile ROW.
__global__ void swSquareKernel(int* H, const int* a, const int* b, int L, int d, int firstRow, int stride) {
    extern __shared__ int swTile[];
    // TODO 1: W, T, and this block's tile (row, column). Do nothing for a tile
    //         outside the table (with a right launch there is none).
    // TODO 2: the 2W - 1 steps inside the tile. In step d_tile this thread's cell is
    //         on its tile row at tile column d_tile - r_tile. Compute it only if it is
    //         inside the tile and inside the table:
    //             max4(0, nw + s, w + SW_GAP, n + SW_GAP)
    //         with s = SW_MATCH when a[r - 1] == b[q - 1], else SW_MISMATCH, and put it
    //         in swTile. Then every thread, whether it computed a cell or not, waits for
    //         the rest of the block before the next step.
    // TODO 3: store the tile.
}

// Fig. 16.8, launching exactly the tiles of each front (the exercise on p. 386).
// Front d holds the tiles (row, d - row). Blocks of W threads, W * (W + pad) ints of
// dynamic shared memory.
void launchSwSquare(int* H, const int* a, const int* b, int Lseq, int W, int pad) {
    // TODO 4: for each of the 2T - 1 fronts, which tile rows does it hold? Launch
    //         one block per tile.
}

// ---------------------------------------------------------------- sheared tiles
// Fig. 16.19: zero the tile, each thread one column, then wait for the block.
__device__ inline void hyInitTile(int* swTile, int W, int stride) {
    // TODO
}

// Fig. 16.17. The sheared tile is stored as a square: in step q_tile, thread r_tile
// writes its cell to swTile[r_tile * stride + q_tile], and that cell is table column
// W * tile_col + q_tile - r_tile + 1. Work out at which stored column each neighbour
// sits, and when it belongs to an earlier tile instead (then it is in H).
__device__ inline int hyLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    // TODO
    return 0;
}
__device__ inline int hyLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    // TODO
    return 0;
}
__device__ inline int hyLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    // TODO
    return 0;
}

// Fig. 16.18: copy the tile to H, the thread index being the stored column. Apply the
// shear, and skip cells outside the table on every side.
__device__ inline void hyStoreTile(int* H, const int* swTile, int L, int W, int stride, int tile_row, int tile_col) {
    // TODO
}

// Fig. 16.16.
__global__ void swHyperKernel(int* H, const int* a, const int* b, int L, int d, int firstRow, int stride) {
    extern __shared__ int swTile[];
    // TODO 5: this block's tile. The row is found as before; each tile row runs two
    //         tile columns behind the row above (p. 392). How far right can a sheared
    //         tile reach and still hold cells of the table?
    // TODO 6: zero the tile.
    // TODO 7: W steps. In step d_tile every thread's cell is at stored column d_tile.
    //         Compute it when it is inside the table, then wait for the block.
    // TODO 8: store the tile.
}

// Fig. 16.15, launching exactly the tiles of each front: 3T - 1 fronts, blocks of W
// threads, W * (W + pad) ints of dynamic shared memory.
void launchSwHyper(int* H, const int* a, const int* b, int Lseq, int W, int pad) {
    // TODO 9: which tile rows does front d hold?
}
