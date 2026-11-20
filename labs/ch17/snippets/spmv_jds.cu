// Chapter 17, summary page: SpMV/JDS, one thread per sorted row. The book shows no JDS
// kernel; exercise 5 (p. 422) asks for one, and this answer is ours, built on the arrays
// of Fig. 17.14 (p. 416). Rows are sorted longest first, as the figure sorts them (the
// text's "increasing order" on p. 416 is a slip: notes/ch17.md "Unclear" item 3).
//  - iterPtr has numIters + 1 entries: iteration t's elements sit at iterPtr[t] up to
//    iterPtr[t+1], one per sorted row that has a t-th element, those rows first.
//  - row[r] is the original row of sorted row r. The result goes to y[row[r]]: sorting
//    the rows permuted y, not x (notes item 5).
//  - y[row[r]] += sum: SpMV accumulates, as Fig. 17.9 does (notes item 8).
//  - The JDSMatrix struct is ours; the field names follow Fig. 17.14.
// Run on the course's CPU emulator by app/src/chapters/ch17/Wrap.snippets.test.ts through
// check/spmv_jds_check.cu.

struct JDSMatrix {
    unsigned int numRows;
    unsigned int numIters;    // the longest row's length
    unsigned int* row;        // original row of each sorted row
    unsigned int* iterPtr;    // numIters + 1 entries
    unsigned int* colIdx;
    float* value;
};

__global__ void spmv_jds_kernel(JDSMatrix jdsMatrix, float* x, float* y) {
    unsigned int r = blockIdx.x*blockDim.x + threadIdx.x;   // a sorted row
    if (r < jdsMatrix.numRows) {
        float sum = 0.0f;
        for (unsigned int t = 0; t < jdsMatrix.numIters && r < jdsMatrix.iterPtr[t + 1] - jdsMatrix.iterPtr[t]; ++t) {
            unsigned int i = jdsMatrix.iterPtr[t] + r;
            unsigned int col = jdsMatrix.colIdx[i];
            float value = jdsMatrix.value[i];
            sum += x[col]*value;
        }
        y[jdsMatrix.row[r]] += sum;
    }
}
