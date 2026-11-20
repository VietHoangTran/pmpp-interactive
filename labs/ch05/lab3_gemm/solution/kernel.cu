// Lab 5.3 reference solution (CUDA; also built for HIP and for the CPU emulator).
// General tiled matrix multiplication with boundary checks:
//   C[j x l] = A[j x k] * B[k x l], row-major, any sizes.
// Book: section 5.5, Fig. 5.13 (p. 115), and the rectangular extension left as
// an exercise on pp. 114-115. First milestone of the through-line project: this
// is the kernel behind every linear layer.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void gemmKernel(const float* A, const float* B, float* C, int j, int k, int l) {
    __shared__ float As[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bs[TILE_WIDTH][TILE_WIDTH];

    int tx = threadIdx.x, ty = threadIdx.y;
    int Row = blockIdx.y * TILE_WIDTH + ty;   // row of C and of A:    valid if < j
    int Col = blockIdx.x * TILE_WIDTH + tx;   // column of C and of B: valid if < l

    float Cvalue = 0;
    // The dot products run over k (width of A = height of B), rounded up to whole tiles.
    for (int ph = 0; ph < (k + TILE_WIDTH - 1) / TILE_WIDTH; ++ph) {
        int aCol = ph * TILE_WIDTH + tx;      // column of A this thread loads
        int bRow = ph * TILE_WIDTH + ty;      // row of B this thread loads

        // Every thread loads for its block even if it owns no valid C element, and
        // every load is checked against the bounds of the array it reads. A slot
        // with no element behind it gets 0, which is harmless in a sum of products.
        As[ty][tx] = (Row < j && aCol < k) ? A[Row * k + aCol] : 0.0f;
        Bs[ty][tx] = (bRow < k && Col < l) ? B[bRow * l + Col] : 0.0f;
        __syncthreads();

        for (int i = 0; i < TILE_WIDTH; ++i) {
            Cvalue += As[ty][i] * Bs[i][tx];
        }
        __syncthreads();
    }
    if (Row < j && Col < l) {
        C[Row * l + Col] = Cvalue;
    }
}

void launchGemm(float* A, float* B, float* C, int j, int k, int l) {
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);
    dim3 dimGrid((l + TILE_WIDTH - 1) / TILE_WIDTH,    // x covers the columns of C
                 (j + TILE_WIDTH - 1) / TILE_WIDTH);   // y covers the rows of C
    gemmKernel<<<dimGrid, dimBlock>>>(A, B, C, j, k, l);
}
