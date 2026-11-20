// Lab 7.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Three ways to hold a convolution filter. Book: sections 7.2 and 7.4,
// Fig. 7.7 (p. 166), Fig. 7.9 (p. 170) and the by-value alternative on p. 172.
//
// Contract for all three kernels: N and P are height x width floats, row-major.
// The filter is (2r+1) x (2r+1) floats, row-major, 0 <= r <= MAX_FILTER_RADIUS,
// applied without flipping (the book's formula, p. 162). Inputs outside the image
// are ghost cells with value 0. One thread per output element, in
// TILE_WIDTH x TILE_WIDTH blocks.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

// The book: "the radius of most convolution filters is 7 or smaller" (p. 168).
#define MAX_FILTER_RADIUS 7
#define MAX_FILTER_DIM (2 * MAX_FILTER_RADIUS + 1)

// @part basic
// Fig. 7.7, corrected. Where this differs from the page:
//   - F and P are float*, so the printed F[fRow][fCol] and P[outRow][outCol] do not
//     compile; both are linearised here (Fig. 7.9 already does it for P).
//   - inRow and inCol are declared int (the listing gives them no type).
//   - The output guard is ours: the printed kernel assumes the grid covers the
//     image exactly, and these blocks overhang any image that is not a multiple
//     of TILE_WIDTH.
__global__ void convBasicKernel(const float* N, const float* F, float* P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;   // a thread past the edge of the image

    float Pvalue = 0.0f;                                // a register accumulator
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;              // the patch's upper-left corner is
            int inCol = outCol - r + fCol;              // (outRow - r, outCol - r)
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {   // skip ghost cells
                Pvalue += F[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

void launchConvBasic(const float* N, const float* F, float* P, int r, int width, int height) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    convBasicKernel<<<dimGrid, dimBlock>>>(N, F, P, r, width, height);
}

// @part const
// Fig. 7.9. Where this differs from the page: the book declares
// F[2*FILTER_RADIUS+1][2*FILTER_RADIUS+1] for one compile-time radius. Here F_c is
// flat and sized for the largest radius, and the first (2r+1)*(2r+1) entries hold
// the filter for the run-time r, so one build serves every radius from 0 to 7.
//
// The symbol, the kernel and the copy are in one file on purpose. The book says a
// kernel in another file needs an extern declaration (p. 169); in nvcc's default
// whole-program mode that is an error and needs -rdc=true (HIP: -fgpu-rdc).
__constant__ float F_c[MAX_FILTER_DIM * MAX_FILTER_DIM];

__global__ void convConstKernel(const float* N, float* P, int r, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;

    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                // Every thread of a warp reads the same F_c element here: the case
                // the constant cache is built for.
                Pvalue += F_c[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

void launchConvConst(const float* N, const float* F_h, float* P, int r, int width, int height) {
    // The third argument is a count of BYTES: (2r+1)^2 floats.
    cudaMemcpyToSymbol(F_c, F_h, (2 * r + 1) * (2 * r + 1) * sizeof(float));
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    convConstKernel<<<dimGrid, dimBlock>>>(N, P, r, width, height);
}

// @part byvalue
// The book's first alternative (p. 172): put the weights in a struct and pass it by
// value, so each launch carries its own filter and no symbol copy is needed. 904
// bytes. A plain const parameter, without CUDA's __grid_constant__: we found no HIP
// documentation of that qualifier, and this file must also build for HIP.
struct Filter {
    int r;
    float w[MAX_FILTER_DIM * MAX_FILTER_DIM];
};

__global__ void convByValueKernel(const float* N, float* P, const Filter f, int width, int height) {
    int outCol = blockIdx.x * blockDim.x + threadIdx.x;
    int outRow = blockIdx.y * blockDim.y + threadIdx.y;
    if (outRow >= height || outCol >= width) return;

    const int r = f.r;
    float Pvalue = 0.0f;
    for (int fRow = 0; fRow < 2 * r + 1; fRow++) {
        for (int fCol = 0; fCol < 2 * r + 1; fCol++) {
            int inRow = outRow - r + fRow;
            int inCol = outCol - r + fCol;
            if (inRow >= 0 && inRow < height && inCol >= 0 && inCol < width) {
                Pvalue += f.w[fRow * (2 * r + 1) + fCol] * N[inRow * width + inCol];
            }
        }
    }
    P[outRow * width + outCol] = Pvalue;
}

void launchConvByValue(const float* N, const float* F_h, float* P, int r, int width, int height) {
    Filter f = {};   // zero the unused tail of w
    f.r = r;
    for (int i = 0; i < (2 * r + 1) * (2 * r + 1); i++) f.w[i] = F_h[i];
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((width + TILE_WIDTH - 1) / TILE_WIDTH, (height + TILE_WIDTH - 1) / TILE_WIDTH);
    convByValueKernel<<<dimGrid, dimBlock>>>(N, P, f, width, height);   // the runtime copies f at launch
}
