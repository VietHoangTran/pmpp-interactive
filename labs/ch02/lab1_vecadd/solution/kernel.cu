// Lab 2.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// The whole program of chapter 2: the kernel of Fig. 2.10 (p. 35) and the host
// function of Fig. 2.13 (p. 38), with the error check of the sidebar on p. 32.
//
// Where this file differs from the page:
//   - p. 32 declares `err` and then tests `error`, which does not compile. The
//     macro below tests `err`. It also returns instead of calling exit(), so the
//     driver can test what happens when a call fails.
//   - Fig. 2.13 returns void, hard-wires 256 threads per block and sizes the
//     grid with ceil(n/256.0). Here vecAdd returns a status, takes the block
//     size as an argument and uses the integer ceiling of the book's Fig. 2.15.
//   - Fig. 2.13 checks nothing and does not ask whether the launch worked. The
//     two lines after the launch are not in the book.
//   - The inputs are const, and the byte count is a size_t (an int overflows
//     once n reaches 2^29 floats).
#include <cstdio>

// The sidebar on p. 32, corrected, as a macro. It reports and makes the
// enclosing function return 1. It frees nothing: after a failure the device
// arrays allocated so far stay allocated.
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// Compute vector sum C = A + B.
// Each thread performs one pair-wise addition.
__global__ void vecAddKernel(const float* A, const float* B, float* C, int n) {
    int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < n) {            // the last block may have threads past the end
        C[i] = A[i] + B[i];
    }
}

int vecAdd(const float* A_h, const float* B_h, float* C_h, int n, int threadsPerBlock) {
    float *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    size_t size = (size_t)n * sizeof(float);    // bytes, not elements

    // Part 1: allocate device memory for A, B and C; copy A and B to the device.
    CHECK(cudaMalloc((void**)&A_d, size));
    CHECK(cudaMalloc((void**)&B_d, size));
    CHECK(cudaMalloc((void**)&C_d, size));

    CHECK(cudaMemcpy(A_d, A_h, size, cudaMemcpyHostToDevice));   // destination first
    CHECK(cudaMemcpy(B_d, B_h, size, cudaMemcpyHostToDevice));

    // Part 2: launch enough whole blocks to cover n elements.
    int numBlocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    vecAddKernel<<<numBlocks, threadsPerBlock>>>(A_d, B_d, C_d, n);
    CHECK(cudaGetLastError());         // was the launch accepted? Reading the state also clears it.
    CHECK(cudaDeviceSynchronize());    // wait for the grid; reports a failure while it ran

    // Part 3: copy C from the device; free the device arrays.
    CHECK(cudaMemcpy(C_h, C_d, size, cudaMemcpyDeviceToHost));

    CHECK(cudaFree(A_d));
    CHECK(cudaFree(B_d));
    CHECK(cudaFree(C_d));
    return 0;
}
