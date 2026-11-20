// Lab 17.1 reference solution (CUDA; also built for HIP and for the CPU emulators).
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
// Where this differs from the page:
//   - The book passes a CSRMatrix or ELLMatrix struct by value. Here each array is its own
//     argument, which maps one-to-one onto Metal's buffer bindings
//     (notes/today/ch17-facts.md A11, C2).
//   - Fig. 17.12 ends with `y[row] = sum;`. SpMV multiplies and accumulates (p. 403),
//     Fig. 17.9 ends with `y[row] += sum;`, and every library accumulates (ch17-facts A10).
//     So does the ELL kernel here; it is also what lets the hybrid run its COO part first
//     (notes/ch17.md, "Unclear in the PDF", item 8).
//   - Fig. 17.12 steps from one slot of a row to the next by numRows. Here by stride, which
//     may be larger: the book's own remark (p. 413) about appending rows before transposing
//     so that every iteration starts on an aligned address, made explicit (ch17-facts A3).
//   - The book does not say what a pad slot's column holds. Here -1, as in cuSPARSE's
//     Sliced ELL and rocSPARSE's ELL (ch17-facts A6, B3). This kernel never reads a pad: it
//     stops at nnzPerRow[row], as Fig. 17.12 does. A kernel that visits all `width` slots
//     instead must test col >= 0 before reading x[col], as rocSPARSE's reference loop does.
//   - The launchers are ours; the book shows no host code.
#include "../spmv.h"   // SPMV_BLOCK

__global__ void spmvCsrKernel(const int* rowPtrs, const int* colIdx, const float* values,
                              const float* x, float* y, int numRows) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {                  // the grid is whole blocks: some threads have no row
        float sum = 0.0f;                 // in a register
        for (int i = rowPtrs[row]; i < rowPtrs[row + 1]; ++i) {
            int col = colIdx[i];
            float value = values[i];
            sum += x[col] * value;
        }
        y[row] += sum;                    // one thread owns each row: no atomic needed
    }
}

void launchSpmvCsr(const int* rowPtrs, const int* colIdx, const float* values,
                   const float* x, float* y, int numRows) {
    // No rows, no launch: HIP rejects a grid of zero blocks, and CUDA's documentation does
    // not state the case (notes/today/ch02-facts.md B4, A8).
    if (numRows == 0) return;
    spmvCsrKernel<<<(numRows + SPMV_BLOCK - 1) / SPMV_BLOCK, SPMV_BLOCK>>>(
        rowPtrs, colIdx, values, x, y, numRows);
}

__global__ void spmvEllKernel(const int* colIdx, const float* values, const int* nnzPerRow,
                              const float* x, float* y, int numRows, int width, int stride) {
    // `width` is not needed when the loop stops at nnzPerRow[row].
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < numRows) {
        float sum = 0.0f;
        for (int t = 0; t < nnzPerRow[row]; ++t) {
            int i = t * stride + row;     // neighbouring threads read neighbouring slots
            int col = colIdx[i];
            float value = values[i];
            sum += x[col] * value;
        }
        y[row] += sum;                    // accumulate, as Fig. 17.9 does (not Fig. 17.12's `=`)
    }
}

void launchSpmvEll(const int* colIdx, const float* values, const int* nnzPerRow,
                   const float* x, float* y, int numRows, int width, int stride) {
    if (numRows == 0) return;
    spmvEllKernel<<<(numRows + SPMV_BLOCK - 1) / SPMV_BLOCK, SPMV_BLOCK>>>(
        colIdx, values, nnzPerRow, x, y, numRows, width, stride);
}
