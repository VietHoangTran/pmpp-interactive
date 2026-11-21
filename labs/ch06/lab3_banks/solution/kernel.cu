// Lab 6.3 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Matrix transpose through a shared-memory tile. Book: section 6.4 (pp. 139-141) for the
// bank conflict and the padding; section 6.1 (pp. 129-130, Fig. 6.4) for turning a tile
// in shared memory so that both global accesses stay coalesced.
//
// in is rows x cols and out is cols x rows, both row-major: out[c * rows + r] = in[r * cols + c].
// One TILE_WIDTH x TILE_WIDTH block per tile. The block reads its input tile row by row
// and writes its output tile row by row, so neighbouring threads touch neighbouring
// global addresses both times. The turn happens in shared memory: each thread writes a
// row of the tile and later reads down a column of it.
//
// pitch is the distance in words between two rows of the tile. With pitch = TILE_WIDTH,
// the words a warp reads down a column are TILE_WIDTH apart; with pitch = TILE_WIDTH + 1
// they are one word further apart each, which is the book's padding (p. 140).
//
// Where this differs from the book: section 6.4 shows only the declarations
// a[TILE_DIM][TILE_DIM] and a[TILE_DIM][TILE_DIM + 1] for a 32 x 32 block, with the
// conflict in the store a[threadIdx.x][threadIdx.y] = .... Here the tile is one flat
// array indexed tile[r * pitch + c], so that one kernel serves both layouts with the pitch
// chosen at run time, and the column access is the read out of the tile, as in the
// transpose of NVIDIA's programming guide (notes/today/ch06-facts.md, A6).
//
// Under NVIDIA's documented layout (32 banks of 4 bytes, warps of 32 consecutive
// threads; infra-facts A10), our arithmetic gives the worst conflict per warp access:
//   TILE_WIDTH 32: read 32-way at pitch 32, none at pitch 33; the write has none.
//   TILE_WIDTH 16: a warp spans two rows of threads. Read 8-way at pitch 16; at pitch 17
//                  both the read and the write are 2-way at one bank.
// AMD's banks and Apple's threadgroup memory follow other rules (ch06 facts B4, C2).
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void tiledTransposeKernel(const float* in, float* out, int rows, int cols, int pitch) {
    // TILE_WIDTH rows of TILE_WIDTH + 1 words, enough for the padded layout; pitch =
    // TILE_WIDTH uses the first TILE_WIDTH^2 words. (Written without parentheses, which the
    // CPU emulator's translator once rejected in a __shared__ array bound; it accepts them now.)
    __shared__ float tile[TILE_WIDTH * TILE_WIDTH + TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;

    // Input tile: rows blockIdx.y * T .., columns blockIdx.x * T ...
    int r = blockIdx.y * TILE_WIDTH + ty;
    int c = blockIdx.x * TILE_WIDTH + tx;
    if (r < rows && c < cols) {
        tile[ty * pitch + tx] = in[r * cols + c];   // a warp reads along a row of in: coalesced
    }
    __syncthreads();   // the whole tile is in shared memory before anyone reads it

    // Output tile: the transposed block. Its rows are input columns, its columns input rows.
    int outRow = blockIdx.x * TILE_WIDTH + ty;
    int outCol = blockIdx.y * TILE_WIDTH + tx;
    if (outRow < cols && outCol < rows) {
        out[outRow * rows + outCol] = tile[tx * pitch + ty];   // reads down a column of the tile
    }
}

void launchTiledTranspose(const float* in, float* out, int rows, int cols, int pitch) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((cols + TILE_WIDTH - 1) / TILE_WIDTH,    // x runs along the input's columns
                 (rows + TILE_WIDTH - 1) / TILE_WIDTH);   // y along its rows
    tiledTransposeKernel<<<dimGrid, dimBlock>>>(in, out, rows, cols, pitch);
}
