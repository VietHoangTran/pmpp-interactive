// Chapter 17, page 17.2: SpMV/CSC, one thread per column (the book's Fig. 17.18, p. 419,
// shown there for completeness). Differs from the page: the field names rowIdx and value,
// where the listing has rowIdxs and values (notes/ch17.md "Unclear" item 7; the text and
// Fig. 17.16 use the singular); the #include the listing leaves out; and the CSCMatrix
// struct, which the book never defines. The page's caption calls it an SpMV/CSR kernel
// ("Unclear" item 6); it is the CSC one. For the CUDA front-end check and the emulators:
// on HIP, <cuda/atomic> needs libhipcxx (notes/today/infra-facts.md B2). Run on the
// course's CPU emulator by app/src/chapters/ch17/snippets-s172.test.ts through
// check/spmv_check.cu.
#include <cuda/atomic>

struct CSCMatrix {
    unsigned int numCols;
    unsigned int* colPtrs;
    unsigned int* rowIdx;
    float* value;
};

__global__ void spmv_csc_kernel(CSCMatrix cscMatrix, float* x, float* y) {
    unsigned int col = blockIdx.x*blockDim.x + threadIdx.x;
    if (col < cscMatrix.numCols) {
        float inValue = x[col];
        for (unsigned int i = cscMatrix.colPtrs[col]; i < cscMatrix.colPtrs[col + 1]; ++i) {
            unsigned int row = cscMatrix.rowIdx[i];
            float value = cscMatrix.value[i];
            cuda::atomic_ref<float, cuda::thread_scope_device> y_ref(y[row]);
            y_ref.fetch_add(inValue*value, cuda::memory_order_relaxed);
        }
    }
}
