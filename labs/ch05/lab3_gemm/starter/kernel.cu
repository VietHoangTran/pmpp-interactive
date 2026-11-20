// Lab 5.3 starter (CUDA; also built for HIP and for the CPU emulator).
// General tiled matrix multiplication:
//   C[j x l] = A[j x k] * B[k x l], row-major, ANY sizes.
// Book: section 5.5 (pp. 112-115). First milestone of the through-line project:
// this is the kernel behind every linear layer (j = batch, k = inputs, l = outputs).
//
// Start from your lab 5.2 kernel. Three things change:
//   1. `Width` meant three different things. Decide which of j, k, l each use becomes.
//   2. Sizes need not be multiples of TILE_WIDTH, so every memory access needs
//      its own bounds check, and the number of phases must round up.
//   3. The grid is no longer square.
#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

__global__ void gemmKernel(const float* A, const float* B, float* C, int j, int k, int l) {
    // TODO
}

void launchGemm(float* A, float* B, float* C, int j, int k, int l) {
    // TODO
}
