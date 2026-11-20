// Lab 17.1, provided (do not edit): SpMV/COO, the second half of the hybrid ELL + COO format.
// Book: section 17.2, Fig. 17.5, p. 405; the hybrid is section 17.5, pp. 414-415.
// One thread per non-zero. Several non-zeros share a row, so several threads add into the
// same y[row], and the add is atomic. The tests run it too, so a broken copy of this file shows.
//
// Contract: rowIdx, colIdx, values [numNonzeros]; x [numCols]; y [numRows], which already
// holds values: this kernel ADDS its entries' products to it. Any entry order.
//
// Where this differs from the page:
//   - Fig. 17.5 takes a COOMatrix struct; here each array is its own argument, as in the
//     rest of the lab (notes/today/ch17-facts.md A11).
//   - The listing shows no #include; cuda::atomic_ref lives in <cuda/atomic> (ch17-facts A1).
//   - The launcher is ours: the book shows no host code. It launches nothing when there are
//     no entries: HIP rejects a grid of zero blocks, and CUDA's documentation does not state
//     the case (notes/today/ch02-facts.md B4, A8).
// The atomic itself is the book's: device scope, relaxed order. On global memory a float
// atomic add flushes subnormals to zero (notes/today/ch10-facts.md A6), and the order of the
// adds into one y[row] is undefined (ch09-facts A3), so the result can differ from a CPU
// reference in the last bits. The tests use a tolerance for that reason.
#include <cuda/atomic>

#include "../spmv.h"

__global__ void spmvCooKernel(const int* rowIdx, const int* colIdx, const float* values,
                              const float* x, float* y, int numNonzeros) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numNonzeros) {
        int row = rowIdx[i];
        int col = colIdx[i];
        float value = values[i];
        cuda::atomic_ref<float, cuda::thread_scope_device> y_ref(y[row]);
        y_ref.fetch_add(x[col] * value, cuda::memory_order_relaxed);
    }
}

void launchSpmvCoo(const int* rowIdx, const int* colIdx, const float* values,
                   const float* x, float* y, int numNonzeros) {
    if (numNonzeros == 0) return;
    spmvCooKernel<<<(numNonzeros + SPMV_BLOCK - 1) / SPMV_BLOCK, SPMV_BLOCK>>>(
        rowIdx, colIdx, values, x, y, numNonzeros);
}
