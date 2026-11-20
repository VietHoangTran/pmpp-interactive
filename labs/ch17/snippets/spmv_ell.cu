// Chapter 17, page 17.2: SpMV/ELL, one thread per row over column-major padded storage
// (the book's Fig. 17.12, p. 412), and the same kernel looping the full padded width.
// Differs from the page:
//  - y[row] += sum, where the page has y[row] = sum (notes/ch17.md "Unclear" item 8): SpMV
//    accumulates, as Fig. 17.9 and every library do (notes/today/ch17-facts.md A10).
//  - A pad slot holds column -1 and value 0, as NVIDIA's Sliced ELL and AMD's ELL do
//    (facts A6, B3); -1 needs a signed index, so colIdx and col are int. The full-width
//    loop tests col >= 0 before it reads x.
//  - stride where the page writes numRows: the two are equal unless whole rows of pads
//    are added so that every iteration starts on an aligned address (p. 413, facts A3).
//  - The ELLMatrix struct, which the book never defines.
// Run on the course's CPU emulator by app/src/chapters/ch17/snippets-s172.test.ts through
// check/spmv_check.cu.

struct ELLMatrix {
    unsigned int numRows;
    unsigned int stride;      // distance between a row's consecutive elements, at least numRows
    unsigned int width;       // slots per row: the longest row's length
    unsigned int* nnzPerRow;
    int* colIdx;              // element t of row r at t*stride + r; -1 in a pad slot
    float* value;             // 0 in a pad slot
};

// The book's loop: each thread stops at its own row's length.
__global__ void spmv_ell_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.nnzPerRow[row]; ++t) {
            unsigned int i = t*ellMatrix.stride + row;
            int col = ellMatrix.colIdx[i];
            float value = ellMatrix.value[i];
            sum += x[col]*value;
        }
        y[row] += sum;
    }
}

// Without nnzPerRow: every thread runs the full width and skips the pads.
__global__ void spmv_ell_full_kernel(ELLMatrix ellMatrix, float* x, float* y) {
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    if (row < ellMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < ellMatrix.width; ++t) {
            unsigned int i = t*ellMatrix.stride + row;
            int col = ellMatrix.colIdx[i];
            if (col >= 0) sum += x[col]*ellMatrix.value[i];
        }
        y[row] += sum;
    }
}
