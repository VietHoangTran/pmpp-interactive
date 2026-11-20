// Lab 2.4 starter: a host stub that almost works.
//
// scalePrefix multiplies the first `len` elements of a host array of `n` floats
// by `s`, on the device, and leaves the padding from len to n as it was: the
// shape of applying an operation to the valid part of a padded batch. Not in
// the book. The kernel is the book's Fig. 2.10 (p. 35) with one array, and the
// host function is its Fig. 2.13 (p. 38) with one array that goes both ways.
//
// This file has FOUR independent defects. It compiles and it runs. Find and
// fix all four.
// Run it on the CPU emulator first, fix one thing, run again.
#include <cstdio>

// The check of the sidebar on p. 32, as in lab 2.1. It reports and makes the
// enclosing function return 1. It frees nothing. (The macro is not a defect.)
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

__global__ void scalePrefixKernel(float* x, float s, int len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    x[i] = s * x[i];
}

// Returns 0 when everything worked and a non-zero value otherwise.
int scalePrefix(float* x_h, int n, int len, float s, int threadsPerBlock) {
    float* x_d = nullptr;
    size_t size = (size_t)n * sizeof(float);    // bytes of the whole array, padding included

    CHECK(cudaMalloc((void**)&x_d, size));
    CHECK(cudaMemcpy(x_d, x_h, n, cudaMemcpyHostToDevice));

    int numBlocks = len / threadsPerBlock;
    scalePrefixKernel<<<numBlocks, threadsPerBlock>>>(x_d, s, len);
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(x_d));
    return 0;
}
