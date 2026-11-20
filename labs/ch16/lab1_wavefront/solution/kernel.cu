// Lab 16.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// The Smith-Waterman scoring table, tiled two ways.
// Book: section 16.6 (pp. 384-389, Figs. 16.8 to 16.12), square tiles;
//       section 16.7 (pp. 389-396, Figs. 16.15 to 16.20), sheared tiles ("hypertiles").
//
// Contract: H is (L_seq + 1) x (L_seq + 1) ints, row-major, with row 0 and column 0
// already zero; the kernels never write them. a indexes rows, b columns. L = L_seq + 1
// is the width of a table row. A tile is W x W cells (W = threads per block); T =
// ceil(L_seq / W) tiles per side. In shared memory, tile element (row, col) is
// swTile[row * stride + col], stride = W + pad: pad = 1 leaves one unused int after
// every tile row.
//
// Where this file differs from the printed listings:
//  - All index arithmetic is int. The book computes `tile_col = d - blockIdx.x` from
//    unsigned operands and relies on the conversion to int giving a negative number,
//    which C++17 leaves to the implementation (notes/today/ch16-facts.md A3).
//  - Each launch covers exactly the tiles of one front: tile row firstRow + blockIdx.x.
//    This is the exercise on p. 386; the book launches numTiles_x blocks every time and
//    lets the ones without a tile exit.
//  - The tile has a row stride, so padding is one gap per tile row (Fig. 16.20). The
//    book's pad() macro inserts a gap every NUM_BANKS ints instead, which is one per row
//    only when W equals the number of banks (notes/ch16.md, "Unclear in the PDF").
//  - The sheared store tests q >= 1 (Fig. 16.18 tests only q < L, with L unsigned, so a
//    negative q fails by wrapping around; with an int L it would write into the row above).
//  - MATCH, MISMATCH, DELETION, INSERTION and the sequence type T are not defined in the
//    listings; here they are SW_MATCH, SW_MISMATCH, SW_GAP (sw.h) and int.
#include <algorithm>
#include <cstddef>

#include "../sw.h"

// Fig. 16.11.
__device__ inline int max4(int a, int b, int c, int d) {
    int m = a;
    if (m < b) m = b;
    if (m < c) m = c;
    if (m < d) m = d;
    return m;
}

// ---------------------------------------------------------------- square tiles
// Fig. 16.10 with a row stride. A neighbour inside the tile comes from shared memory;
// one on the tile's top row or left column belongs to a tile an earlier launch stored.
__device__ inline int sqLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (r_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + q_tile];
}
__device__ inline int sqLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
__device__ inline int sqLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + (q - 1)] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}

// Fig. 16.12. Here the thread index is the tile COLUMN: in each iteration the block
// writes one contiguous piece of a table row, so the stores are coalesced (p. 389).
__device__ inline void sqStoreTile(int* H, const int* swTile, int L, int W, int stride, int tile_row, int tile_col) {
    const int tx = (int)threadIdx.x;
    for (int row = 0; row < W; row++) {
        const int r = W * tile_row + row + 1;
        const int q = W * tile_col + tx + 1;
        if (r < L && q < L) H[r * L + q] = swTile[row * stride + tx];
    }
}

// Fig. 16.9. One block per tile; the thread index is the tile ROW.
__global__ void swSquareKernel(int* H, const int* a, const int* b, int L, int d, int firstRow, int stride) {
    extern __shared__ int swTile[];
    const int W = (int)blockDim.x;
    const int numTiles = (L - 1 + W - 1) / W;
    const int tile_row = firstRow + (int)blockIdx.x;
    const int tile_col = d - tile_row;
    if (tile_col >= 0 && tile_col < numTiles) {          // always true with an exact launch
        const int r_tile = (int)threadIdx.x;
        const int r = W * tile_row + r_tile + 1;         // + 1: row 0 is the boundary
        for (int d_tile = 0; d_tile < 2 * W - 1; d_tile++) {
            const int q_tile = d_tile - r_tile;
            const int q = W * tile_col + q_tile + 1;
            if (q_tile >= 0 && q_tile < W && r < L && q < L) {
                const int n = sqLoadN(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int w = sqLoadW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int nw = sqLoadNW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int s = (a[r - 1] == b[q - 1]) ? SW_MATCH : SW_MISMATCH;
                swTile[r_tile * stride + q_tile] = max4(0, nw + s, w + SW_GAP, n + SW_GAP);
            }
            __syncthreads();   // every thread, active in this step or not
        }
        sqStoreTile(H, swTile, L, W, stride, tile_row, tile_col);
    }
}

// Fig. 16.8, with the exact number of blocks per front. Front d holds the tiles
// (row, d - row) with both coordinates in 0 .. T - 1.
void launchSwSquare(int* H, const int* a, const int* b, int Lseq, int W, int pad) {
    const int L = Lseq + 1;
    const int T = (Lseq + W - 1) / W;
    const int stride = W + pad;
    const size_t sharedBytes = (size_t)W * stride * sizeof(int);
    for (int d = 0; d < 2 * T - 1; d++) {
        const int firstRow = std::max(0, d - (T - 1));
        const int lastRow = std::min(d, T - 1);
        swSquareKernel<<<lastRow - firstRow + 1, W, sharedBytes>>>(H, a, b, L, d, firstRow, stride);
    }
}

// ---------------------------------------------------------------- sheared tiles
// Fig. 16.19. Each thread zeroes one column. Slots for cells outside the table are
// never computed but are read as neighbours; zero is also the boundary value.
__device__ inline void hyInitTile(int* swTile, int W, int stride) {
    const int tx = (int)threadIdx.x;
    for (int row = 0; row < W; row++) swTile[row * stride + tx] = 0;
    __syncthreads();
}

// Fig. 16.17. Stored element (r_tile, q_tile) is table column W*tile_col + q_tile -
// r_tile + 1, so the north neighbour (one row up, same table column) is stored at
// column q_tile - 1, the west one at q_tile - 1, the north-west one at q_tile - 2.
// A stored column below 0 belongs to an earlier tile.
__device__ inline int hyLoadN(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (r_tile == 0 || q_tile == 0) ? H[(r - 1) * L + q] : swTile[(r_tile - 1) * stride + (q_tile - 1)];
}
__device__ inline int hyLoadW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                              int stride) {
    return (q_tile == 0) ? H[r * L + (q - 1)] : swTile[r_tile * stride + (q_tile - 1)];
}
__device__ inline int hyLoadNW(const int* H, int r, int q, int L, const int* swTile, int r_tile, int q_tile,
                               int stride) {
    return (r_tile == 0 || q_tile == 0 || q_tile == 1) ? H[(r - 1) * L + (q - 1)]
                                                       : swTile[(r_tile - 1) * stride + (q_tile - 2)];
}

// Fig. 16.18, with q >= 1 tested explicitly (see the top of the file).
__device__ inline void hyStoreTile(int* H, const int* swTile, int L, int W, int stride, int tile_row, int tile_col) {
    const int tx = (int)threadIdx.x;
    for (int row = 0; row < W; row++) {
        const int r = W * tile_row + row + 1;
        const int q = W * tile_col + tx - row + 1;   // the shear, m = -1
        if (r < L && q >= 1 && q < L) H[r * L + q] = swTile[row * stride + tx];
    }
}

// Fig. 16.16. Every step is a full anti-diagonal of W cells, one per thread.
__global__ void swHyperKernel(int* H, const int* a, const int* b, int L, int d, int firstRow, int stride) {
    extern __shared__ int swTile[];
    const int W = (int)blockDim.x;
    const int numTiles = (L - 1 + W - 1) / W;
    const int tile_row = firstRow + (int)blockIdx.x;
    const int tile_col = d - 2 * tile_row;               // two tile columns behind the row above
    if (tile_col >= 0 && tile_col <= numTiles) {         // <=: one incomplete tile past the last column
        hyInitTile(swTile, W, stride);
        const int r_tile = (int)threadIdx.x;
        const int r = W * tile_row + r_tile + 1;
        for (int d_tile = 0; d_tile < W; d_tile++) {
            const int q_tile = d_tile;
            const int q = W * tile_col + q_tile - r_tile + 1;   // the book's _shear(q_tile, r_tile)
            if (r < L && q >= 1 && q < L) {
                const int n = hyLoadN(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int w = hyLoadW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int nw = hyLoadNW(H, r, q, L, swTile, r_tile, q_tile, stride);
                const int s = (a[r - 1] == b[q - 1]) ? SW_MATCH : SW_MISMATCH;
                swTile[r_tile * stride + q_tile] = max4(0, nw + s, w + SW_GAP, n + SW_GAP);
            }
            __syncthreads();
        }
        hyStoreTile(H, swTile, L, W, stride, tile_row, tile_col);
    }
}

// Fig. 16.15, with the exact number of blocks per front. Front d holds the tiles
// (row, d - 2 * row) with 0 <= row <= T - 1 and 0 <= d - 2 * row <= T: tile rows
// ceil((d - T) / 2) .. floor(d / 2), clipped to 0 .. T - 1. 3T - 1 fronts.
void launchSwHyper(int* H, const int* a, const int* b, int Lseq, int W, int pad) {
    const int L = Lseq + 1;
    const int T = (Lseq + W - 1) / W;
    const int stride = W + pad;
    const size_t sharedBytes = (size_t)W * stride * sizeof(int);
    for (int d = 0; d < 3 * T - 1; d++) {
        const int firstRow = std::max(0, (d - T + 1) / 2);   // ceil((d - T) / 2) when d >= T
        const int lastRow = std::min(T - 1, d / 2);
        swHyperKernel<<<lastRow - firstRow + 1, W, sharedBytes>>>(H, a, b, L, d, firstRow, stride);
    }
}
