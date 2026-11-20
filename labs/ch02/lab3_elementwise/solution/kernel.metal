// Lab 2.3 reference solution (Metal Shading Language). Not in the book.
//
// CUDA                                   Metal
// __device__ float f(...)                a plain function: every function in a .metal file is GPU code
// __host__ __device__ float f(...)       no counterpart (see the README)
// float alpha (an argument by value)     constant float& alpha [[buffer(index)]], sent with setBytes
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// if (i < n)                             not needed: launch.mm dispatches exactly n threads
#include <metal_stdlib>
using namespace metal;

// A helper the kernels call. It has no `kernel` attribute, so the app cannot
// see it and only GPU code can call it: the role of a __device__ function.
static inline float scale_shift(float x, float alpha, float beta) {
    return alpha * x + beta;
}

// Y[r][j] = X[r][j] + bias[j]. One thread per element of the rows x d tensor.
kernel void add_bias(device float*       Y    [[buffer(0)]],
                     device const float* X    [[buffer(1)]],
                     device const float* bias [[buffer(2)]],
                     constant uint&      d    [[buffer(3)]],
                     uint i [[thread_position_in_grid]]) {
    // Row-major means i = r * d + j, so the column is i % d.
    Y[i] = X[i] + bias[i % d];
}

// Y[i] = alpha * X[i] + beta.
kernel void scale_shift_k(device float*       Y     [[buffer(0)]],
                          device const float* X     [[buffer(1)]],
                          constant float&     alpha [[buffer(2)]],
                          constant float&     beta  [[buffer(3)]],
                          uint i [[thread_position_in_grid]]) {
    Y[i] = scale_shift(X[i], alpha, beta);
}

// X[i] += R[i], in place.
kernel void add_residual(device float*       X [[buffer(0)]],
                         device const float* R [[buffer(1)]],
                         uint i [[thread_position_in_grid]]) {
    X[i] = X[i] + R[i];
}
