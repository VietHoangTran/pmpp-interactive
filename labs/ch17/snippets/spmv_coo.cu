// Chapter 17, page 17.2: SpMV/COO, one thread per non-zero (the book's Fig. 17.5, p. 405).
// Differs from the page in two places: the #include the listing leaves out, and the
// COOMatrix struct, which the book never defines (the field names are the listing's).
// For the CUDA front-end check and the emulators: on HIP, <cuda/atomic> needs libhipcxx
// (notes/today/infra-facts.md B2). Run on the course's CPU emulator by
// app/src/chapters/ch17/snippets-s172.test.ts through check/spmv_check.cu.
#include <cuda/atomic>

struct COOMatrix {
    unsigned int numNonzeros;
    unsigned int* rowIdx;
    unsigned int* colIdx;
    float* value;
};

__global__ void spmv_coo_kernel(COOMatrix cooMatrix, float* x, float* y) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < cooMatrix.numNonzeros) {
        unsigned int row = cooMatrix.rowIdx[i];
        unsigned int col = cooMatrix.colIdx[i];
        float value = cooMatrix.value[i];
        cuda::atomic_ref<float, cuda::thread_scope_device> y_ref(y[row]);
        y_ref.fetch_add(x[col]*value, cuda::memory_order_relaxed);
    }
}
