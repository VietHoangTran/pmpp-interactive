// Lab 22.2 reference solution (Metal Shading Language). Not in the book.
// The embedding gather of kernel.cu. The other kernels of the step are the earlier labs'
// own kernel.metal files, which the driver compiles.
//
// CUDA                                   Metal
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// if (i < batch * d)                     not needed: launch.mm dispatches exactly batch * d threads
// int d (an argument by value)           constant uint& d [[buffer(3)]], sent with setBytes
#include <metal_stdlib>
using namespace metal;

// H[r][j] = E[tokens[r]][j]. One thread per element of the batch x d output: thread i owns
// H[i] and gathers the one input it needs (output-centric, section 22.2).
// The largest index, tokens[r] * d + j, is below vocab * d: it fits a uint while the
// embedding has fewer than 2^32 floats, which holds for every model this lab builds.
kernel void embed_rows(device const float* E      [[buffer(0)]],
                       device const int*   tokens [[buffer(1)]],
                       device float*       H      [[buffer(2)]],
                       constant uint&      d      [[buffer(3)]],
                       uint i [[thread_position_in_grid]]) {
    uint r = i / d;   // the sequence
    uint j = i % d;   // the feature
    H[i] = E[uint(tokens[r]) * d + j];
}
