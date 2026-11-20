// Chapter 17, page 17.2: SpMV/CSR, one thread per row (the book's Fig. 17.9, p. 408).
// Differs from the page only by the CSRMatrix struct, which the book never defines (the
// field names are the listing's). rowPtrs has numRows + 1 entries: the last one ends the
// last row. Run on the course's CPU emulator by
// app/src/chapters/ch17/snippets-s172.test.ts through check/spmv_check.cu.

struct CSRMatrix {
    unsigned int numRows;
    unsigned int* rowPtrs;
    unsigned int* colIdx;
    float* value;
};

__global__ void spmv_csr_kernel(CSRMatrix csrMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < csrMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int i = csrMatrix.rowPtrs[row]; i < csrMatrix.rowPtrs[row + 1]; ++i) {
            unsigned int col = csrMatrix.colIdx[i];
            float value = csrMatrix.value[i];
            sum += x[col]*value;
        }
        y[row] += sum;
    }
}
