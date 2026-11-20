// Lab 7.2, provided: the baseline the tiled kernels are measured against. Do not edit.
// (CUDA; also built for HIP and for the CPU emulator.)
//
// Book Fig. 7.9 (p. 170): one thread per output, the filter in constant memory,
// every input element read from global memory. Two changes from the page, so that
// the comparison with your tiled kernels is fair:
//   - the radius is a compile-time template parameter, as in your kernels, so the
//     filter loops have the same fixed trip counts in all three;
//   - the filter is a flat array indexed fRow * (2r+1) + fCol, and the symbol is
//     F_base, not F, so it cannot clash with your F_c.
// It also has the output guard Fig. 7.9 leaves out, and declares inRow and inCol.
// Blocks are always 16 x 16, whatever TILE_WIDTH the lab is built with, so every
// build times the same baseline.

#define BASE_MAX_RADIUS 7
#define BASE_MAX_DIM (2 * BASE_MAX_RADIUS + 1)
#define BASE_BLOCK 16

__constant__ float F_base[BASE_MAX_DIM * BASE_MAX_DIM];

template <int FILTER_RADIUS>
__global__ void convConstBaseKernel(const float* N, float* P, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++) {
            int inRow = outRow - FILTER_RADIUS + fRow;
            int inCol = outCol - FILTER_RADIUS + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += F_base[fRow * (2 * FILTER_RADIUS + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

template <int FILTER_RADIUS>
static void launchBase(const float* N, float* P, int width, int height) {
    dim3 dimBlock(BASE_BLOCK, BASE_BLOCK);
    dim3 dimGrid((width + BASE_BLOCK - 1) / BASE_BLOCK, (height + BASE_BLOCK - 1) / BASE_BLOCK);
    convConstBaseKernel<FILTER_RADIUS><<<dimGrid, dimBlock>>>(N, P, width, height);
}

// N and P are device pointers; F_h is the (2r+1) x (2r+1) filter in host memory.
void launchConvConstBaseline(const float* N, const float* F_h, float* P, int r, int width, int height) {
    if (r < 1 || r > BASE_MAX_RADIUS) return;   // the driver asks for 1 to 7 only
    cudaMemcpyToSymbol(F_base, F_h, (2 * r + 1) * (2 * r + 1) * sizeof(float));
    switch (r) {
        case 1: launchBase<1>(N, P, width, height); break;
        case 2: launchBase<2>(N, P, width, height); break;
        case 3: launchBase<3>(N, P, width, height); break;
        case 4: launchBase<4>(N, P, width, height); break;
        case 5: launchBase<5>(N, P, width, height); break;
        case 6: launchBase<6>(N, P, width, height); break;
        case 7: launchBase<7>(N, P, width, height); break;
    }
}
