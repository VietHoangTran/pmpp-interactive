// Lab 7.2 starter (CUDA; also built for HIP and for the CPU emulator).
// Tiled 2D convolution, two ways. Book: section 7.5 (pp. 172-177) and section 7.6
// (pp. 178-179).
//
// Contract: N and P are width x height images of floats, row-major, in device
// memory. Ghost cells (inputs outside the image) count as 0. The filter is
// (2r+1) x (2r+1), applied without flipping, as in the book (p. 162).
//
// Part A: write the halo-tiled kernel and its launch (after Fig. 7.12, p. 174).
// Part B: repair the cached-halo kernel (after Fig. 7.15, p. 178). It compiles and
//         has two defects. One of them is printed in the book.
//
// The build defines TILE_WIDTH (32 by default; 8 and 16 are also built for the
// benchmark), so do not hard-code 32. The radius is a template parameter,
// FILTER_RADIUS, chosen at run time by the provided switch at the bottom of the file.
// Compared with the pages: the constant array is F_c in declaration and use (the book
// declares F_c and reads F), it is flat (15 x 15 floats, for the largest radius) and
// indexed fRow * (2*FILTER_RADIUS+1) + fCol, N_s has its element type, and N is const.
#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// The filter, filled by the provided launch functions with cudaMemcpyToSymbol.
__constant__ float F_c[MAX_FILTER_DIM * MAX_FILTER_DIM];

// @part halo
// Part A: the block is the input tile, one thread per input-tile element.
// OUT_TILE_DIM uses the template parameter FILTER_RADIUS, so it is only meaningful
// inside the templates below.
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

template <int FILTER_RADIUS>
__global__ void convTiledHaloKernel(const float* N, float* P, int width, int height) {
    // TODO A1: the row and column, in the image, of the input element this thread
    //          loads. Consecutive blocks start OUT_TILE_DIM apart, but each block is
    //          IN_TILE_DIM wide: where does its first thread sit? For a thread that
    //          computes, the same row and column name its output element.
    //          threadIdx and blockIdx are unsigned: copy them to int before you
    //          subtract anything from them.

    // TODO A2: declare the shared input tile and load this thread's element into it.
    //          An element outside the image is a ghost cell.

    // TODO A3: no thread may read the tile before every thread has written its slot.

    // TODO A4: compute. Which threads have an output? (Not the ones whose output lies
    //          outside the image, and not the outer FILTER_RADIUS ring of the block.)
    //          Each of them adds up F_c times the input-tile elements of its patch in
    //          a register, then writes P once.
}

template <int FILTER_RADIUS>
void launchHalo(const float* N, float* P, int width, int height) {
    // TODO A5: the block and the grid. How many threads does a block have, and how
    //          many blocks cover a width x height image when each block produces one
    //          output tile? Then launch convTiledHaloKernel<FILTER_RADIUS>.
}

// @part cached
// Part B: repair this kernel. Shared memory holds only the tile itself; halo taps are
// read from global memory, where the book expects them to hit the L2 cache.
#define TILE_DIM TILE_WIDTH

template <int FILTER_RADIUS>
__global__ void convCachedKernel(const float* N, float* P, int width, int height) {
    int col = blockIdx.x * TILE_DIM + threadIdx.x;
    int row = blockIdx.y * TILE_DIM + threadIdx.y;

    // loading input tile
    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row < height) {
        N_s[threadIdx.y][threadIdx.x] = N[row * width + col];
    } else {
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    // calculating output elements
    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                if (threadIdx.x - FILTER_RADIUS + fCol >= 0 &&
                    threadIdx.x - FILTER_RADIUS + fCol < TILE_DIM &&
                    threadIdx.y - FILTER_RADIUS + fRow >= 0 &&
                    threadIdx.y - FILTER_RADIUS + fRow < TILE_DIM) {
                    Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N_s[threadIdx.y + fRow][threadIdx.x + fCol];
                } else {
                    if (row - FILTER_RADIUS + fRow >= 0 &&
                        row - FILTER_RADIUS + fRow < height &&
                        col - FILTER_RADIUS + fCol >= 0 &&
                        col - FILTER_RADIUS + fCol < width) {
                        Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                                  N[(row - FILTER_RADIUS + fRow) * width + col - FILTER_RADIUS + fCol];
                    }
                }
            }
        }
        P[row * width + col] = Pvalue;
    }
}

template <int FILTER_RADIUS>
void launchCached(const float* N, float* P, int width, int height) {
    dim3 dimBlock(TILE_DIM, TILE_DIM);
    dim3 dimGrid((width + TILE_DIM - 1) / TILE_DIM, (height + TILE_DIM - 1) / TILE_DIM);
    convCachedKernel<FILTER_RADIUS><<<dimGrid, dimBlock>>>(N, P, width, height);
}

// @part provided
// Provided: the functions the driver calls. N and P are device pointers; F_h is the
// (2r+1) x (2r+1) filter in host memory. The radius picks a compiled kernel, the
// way llama.cpp's ssm_conv picks its filter width (notes/today/ch07-facts.md D3).

// A radius whose output tile would be empty at this TILE_WIDTH is never compiled.
template <int R>
static void haloIfItFits(const float* N, float* P, int width, int height) {
    if constexpr (IN_TILE_DIM - 2 * R >= 1) launchHalo<R>(N, P, width, height);
}

void launchConvTiledHalo(const float* N, const float* F_h, float* P, int r, int width, int height) {
    if (r < 1 || r > MAX_FILTER_RADIUS || IN_TILE_DIM - 2 * r < 1) return;   // the driver never asks
    cudaMemcpyToSymbol(F_c, F_h, (2 * r + 1) * (2 * r + 1) * sizeof(float));
    switch (r) {
        case 1: haloIfItFits<1>(N, P, width, height); break;
        case 2: haloIfItFits<2>(N, P, width, height); break;
        case 3: haloIfItFits<3>(N, P, width, height); break;
        case 4: haloIfItFits<4>(N, P, width, height); break;
        case 5: haloIfItFits<5>(N, P, width, height); break;
        case 6: haloIfItFits<6>(N, P, width, height); break;
        case 7: haloIfItFits<7>(N, P, width, height); break;
    }
}

void launchConvTiledCached(const float* N, const float* F_h, float* P, int r, int width, int height) {
    if (r < 1 || r > MAX_FILTER_RADIUS) return;   // the driver never asks
    cudaMemcpyToSymbol(F_c, F_h, (2 * r + 1) * (2 * r + 1) * sizeof(float));
    switch (r) {
        case 1: launchCached<1>(N, P, width, height); break;
        case 2: launchCached<2>(N, P, width, height); break;
        case 3: launchCached<3>(N, P, width, height); break;
        case 4: launchCached<4>(N, P, width, height); break;
        case 5: launchCached<5>(N, P, width, height); break;
        case 6: launchCached<6>(N, P, width, height); break;
        case 7: launchCached<7>(N, P, width, height); break;
    }
}
