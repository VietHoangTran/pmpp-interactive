// Lab 7.2 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Tiled 2D convolution, two ways. Book: section 7.5, Fig. 7.12 (p. 174), and
// section 7.6, Fig. 7.15 (p. 178).
//
// Contract: N and P are width x height images of floats, row-major, in device
// memory. Ghost cells (inputs outside the image) count as 0. The filter is
// (2r+1) x (2r+1), applied without flipping, as in the book (p. 162).
//
// Where this file differs from the pages:
//   - The radius is a template parameter, FILTER_RADIUS, chosen at run time by the
//     switch at the bottom of the file. The book fixes it with one #define.
//   - The constant array is F_c in both declaration and use (the book declares F_c
//     and reads F), and it is flat, 15 x 15 floats for the largest radius, indexed
//     fRow * (2*FILTER_RADIUS+1) + fCol.
//   - N_s has its element type (the book's `__shared__ N_s[...]` has none), literals
//     are 0.0f, and N is const.
//   - Thread indices are copied to int before anything is subtracted from them.
//     threadIdx.x - FILTER_RADIUS is unsigned in C++, so a test such as `>= 0` on it
//     is always true (notes/today/ch07-facts.md A8).
//   - Fig. 7.15 line 21 indexes N_s[threadIdx.y+fRow][threadIdx.x+fCol]. The tap it
//     has just tested is N_s[ty-FILTER_RADIUS+fRow][tx-FILTER_RADIUS+fCol], which is
//     what this file reads.
#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// The filter, filled by the provided launch functions with cudaMemcpyToSymbol.
__constant__ float F_c[MAX_FILTER_DIM * MAX_FILTER_DIM];

// @part halo
// Part A: the block is the input tile. OUT_TILE_DIM uses the template parameter
// FILTER_RADIUS, so it is only meaningful inside the templates below.
#define IN_TILE_DIM TILE_WIDTH
#define OUT_TILE_DIM ((IN_TILE_DIM) - 2 * (FILTER_RADIUS))

template <int FILTER_RADIUS>
__global__ void convTiledHaloKernel(const float* N, float* P, int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y;   // int, so that tx - FILTER_RADIUS can be negative
    int bx = blockIdx.x, by = blockIdx.y;
    // The input element this thread loads. For an active thread it is also the
    // output element it computes.
    int col = bx * OUT_TILE_DIM + tx - FILTER_RADIUS;
    int row = by * OUT_TILE_DIM + ty - FILTER_RADIUS;

    // Load phase: every thread loads one element of the input tile.
    __shared__ float N_s[IN_TILE_DIM][IN_TILE_DIM];
    if (row >= 0 && row < height && col >= 0 && col < width) {
        N_s[ty][tx] = N[row * width + col];
    } else {
        N_s[ty][tx] = 0.0f;   // ghost cell, stored so the compute loop needs no test
    }
    __syncthreads();

    // Compute phase: the outer FILTER_RADIUS ring of threads only loaded.
    int tileCol = tx - FILTER_RADIUS;
    int tileRow = ty - FILTER_RADIUS;
    if (col >= 0 && col < width && row >= 0 && row < height) {
        if (tileCol >= 0 && tileCol < OUT_TILE_DIM && tileRow >= 0 && tileRow < OUT_TILE_DIM) {
            float Pvalue = 0.0f;
            for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
                for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                    Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N_s[tileRow + fRow][tileCol + fCol];
                }
            }
            P[row * width + col] = Pvalue;
        }
    }
}

template <int FILTER_RADIUS>
void launchHalo(const float* N, float* P, int width, int height) {
    // One thread per input-tile element; one block per output tile.
    dim3 dimBlock(IN_TILE_DIM, IN_TILE_DIM);
    dim3 dimGrid((width + OUT_TILE_DIM - 1) / OUT_TILE_DIM, (height + OUT_TILE_DIM - 1) / OUT_TILE_DIM);
    convTiledHaloKernel<FILTER_RADIUS><<<dimGrid, dimBlock>>>(N, P, width, height);
}

// @part cached
// Part B: shared memory holds only the tile itself; halo taps are read from global
// memory, where the book expects them to hit the L2 cache (section 7.6).
#define TILE_DIM TILE_WIDTH

template <int FILTER_RADIUS>
__global__ void convCachedKernel(const float* N, float* P, int width, int height) {
    int tx = threadIdx.x, ty = threadIdx.y;
    int col = blockIdx.x * TILE_DIM + tx;
    int row = blockIdx.y * TILE_DIM + ty;

    __shared__ float N_s[TILE_DIM][TILE_DIM];
    if (row < height && col < width) {
        N_s[ty][tx] = N[row * width + col];
    } else {
        N_s[ty][tx] = 0.0f;
    }
    __syncthreads();

    if (col < width && row < height) {
        float Pvalue = 0.0f;
        for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
            for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
                if (tx - FILTER_RADIUS + fCol >= 0 &&
                    tx - FILTER_RADIUS + fCol < TILE_DIM &&
                    ty - FILTER_RADIUS + fRow >= 0 &&
                    ty - FILTER_RADIUS + fRow < TILE_DIM) {
                    // The tap is inside this block's tile.
                    Pvalue += F_c[fRow * (2 * FILTER_RADIUS + 1) + fCol] *
                              N_s[ty - FILTER_RADIUS + fRow][tx - FILTER_RADIUS + fCol];
                } else {
                    // A halo tap: read it from global memory, unless it is a ghost cell.
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
