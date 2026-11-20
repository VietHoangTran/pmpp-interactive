// Lab 17.1 starter (CUDA; also built for HIP and for the CPU emulators).
// SpMV, y = A·x + y, one thread per row, in two formats.
// Book: sections 17.3 and 17.4, pp. 407-414, Figs. 17.9 (CSR) and 17.12 (ELL).
//
// Contract (raw device pointers, explicit sizes; A has numRows rows and numCols columns):
//   CSR  rowPtrs [numRows + 1]; colIdx, values [nnz]. Row r's entries sit at positions
//        rowPtrs[r] .. rowPtrs[r+1] - 1, in any column order.
//   ELL  colIdx, values [width * stride], column-major: slot t of row r is at
//        t * stride + r, with stride >= numRows and width * stride < 2^31.
//        nnzPerRow [numRows]: row r's entries fill its first nnzPerRow[r] slots.
//        The other slots are pads: column -1, value 0.
//   x [numCols]. y [numRows] already holds values: both kernels ADD A·x to it.
//   numRows may be 0; then nothing is launched.
//
// The book passes a struct holding these arrays; here each array is its own argument.
// Two places where this lab asks for something other than the page: read "Where this lab
// differs from the page" in README.md before you start.
#include "../spmv.h"   // SPMV_BLOCK: threads per block, for both launchers

__global__ void spmvCsrKernel(const int* rowPtrs, const int* colIdx, const float* values,
                              const float* x, float* y, int numRows) {
    // TODO 1: which row does this thread own? The launcher rounds the grid up to whole
    //         blocks, so some threads have no row: they must touch nothing.

    // TODO 2: the dot product of this row with x, summed in a register.

    // TODO 3: add the result to y[row] (y already holds a value).
}

void launchSpmvCsr(const int* rowPtrs, const int* colIdx, const float* values,
                   const float* x, float* y, int numRows) {
    // TODO 4: blocks of SPMV_BLOCK threads, enough blocks to cover every row.
    //         When numRows is 0 there is nothing to launch, and HIP rejects a grid of zero
    //         blocks.
}

__global__ void spmvEllKernel(const int* colIdx, const float* values, const int* nnzPerRow,
                              const float* x, float* y, int numRows, int width, int stride) {
    // TODO 5: as TODO 1.

    // TODO 6: walk this row's slots. Two correct ways to know where to stop: the book's
    //         (Fig. 17.12), or the libraries' (visit every slot and skip the pads). Either
    //         way, never read x at a pad's column.

    // TODO 7: as TODO 3. In the hybrid, the provided COO kernel runs before this one.
}

void launchSpmvEll(const int* colIdx, const float* values, const int* nnzPerRow,
                   const float* x, float* y, int numRows, int width, int stride) {
    // TODO 8: as TODO 4.
}
