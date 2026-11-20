// Lab 2.4 reference solution: the four defects of the starter, fixed.
//
// scalePrefix multiplies the first `len` elements of a host array of `n` floats
// by `s`, on the device, and leaves the padding from len to n as it was: the
// shape of applying an operation to the valid part of a padded batch. Not in
// the book. The kernel is the book's Fig. 2.10 (p. 35) with one array, and the
// host function is its Fig. 2.13 (p. 38) with one array that goes both ways.
//
// Where this file differs from the page:
//   - the block count is the integer ceiling of Fig. 2.15 (p. 42), not the
//     ceil(n/256.0) of Fig. 2.13, and the block size is an argument;
//   - every runtime call is checked with the sidebar's pattern (p. 32), which
//     as printed declares `err` and tests `error`; the macro tests `err`;
//   - the launch is followed by cudaGetLastError and a checked
//     cudaDeviceSynchronize. Fig. 2.13 has neither (ch02-facts A6).
#include <cstdio>

// The check of the sidebar on p. 32, as in lab 2.1. It reports and makes the
// enclosing function return 1. It frees nothing.
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
    // FIX 1: the guard. The grid is whole blocks, so it has up to
    // threadsPerBlock - 1 threads more than len. The starter let them run:
    // they scaled the padding, and when len was n they wrote past the end of
    // the device array.
    if (i < len) {
        x[i] = s * x[i];
    }
}

// Returns 0 when everything worked and a non-zero value otherwise.
int scalePrefix(float* x_h, int n, int len, float s, int threadsPerBlock) {
    float* x_d = nullptr;
    size_t size = (size_t)n * sizeof(float);    // bytes of the whole array, padding included

    CHECK(cudaMalloc((void**)&x_d, size));
    // FIX 3: the third argument is a number of bytes. The starter passed n, so
    // only the first quarter of the array reached the device and the kernel
    // worked on whatever the rest of the allocation held.
    CHECK(cudaMemcpy(x_d, x_h, size, cudaMemcpyHostToDevice));

    // FIX 2: round the block count up. The starter wrote len / threadsPerBlock,
    // which rounds down: the last len % threadsPerBlock elements got no thread,
    // and a prefix shorter than one block got no blocks at all.
    int numBlocks = (len + threadsPerBlock - 1) / threadsPerBlock;
    scalePrefixKernel<<<numBlocks, threadsPerBlock>>>(x_d, s, len);
    // FIX 4: the launch statement has no value. The starter went straight on
    // to cudaDeviceSynchronize. On the emulator that call says nothing about a
    // refused launch, so the starter returned 0 for it. Whether CUDA's or HIP's
    // synchronise reports one is not stated in the documentation we checked;
    // the vendors' pattern is to read the error state right after the launch.
    CHECK(cudaGetLastError());
    CHECK(cudaDeviceSynchronize());

    CHECK(cudaMemcpy(x_h, x_d, size, cudaMemcpyDeviceToHost));
    CHECK(cudaFree(x_d));
    return 0;
}
