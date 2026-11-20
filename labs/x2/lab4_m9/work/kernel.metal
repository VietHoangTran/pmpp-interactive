// Lab X2.4 starter (Metal Shading Language). Not in the book.
// The embedding gathers of kernel.cu, one per weight format. The other kernels of the step
// are the earlier labs' own kernel.metal files, which the driver compiles.
//
// CUDA                                   Metal
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// if (i < batch * d)                     not needed if launch.mm dispatches exactly batch * d threads
// halfToFloat(bits), by hand             float(as_type<half>(bits)): half is a core MSL type
//                                        (infra-facts C8), as_type reinterprets bits (x2-facts C5)
// const BlockQ4_0*                       device const block_q4_0*, the same 18-byte layout
#include <metal_stdlib>
using namespace metal;

struct block_q4_0 {          // llama.cpp's block_q4_0 (x2-facts D1): 18 bytes, alignment 2
    ushort d;                // FP16 bits of the scale
    uchar qs[16];            // byte k: weight k (low nibble), weight k + 16 (high nibble)
};

// Provided: lab 22.2's embed_rows, unchanged.
kernel void embed_rows_f32(device const float* E      [[buffer(0)]],
                           device const int*   tokens [[buffer(1)]],
                           device float*       H      [[buffer(2)]],
                           constant uint&      d      [[buffer(3)]],
                           uint i [[thread_position_in_grid]]) {
    const uint r = i / d, j = i % d;
    H[i] = E[uint(tokens[r]) * d + j];
}

// TODO 1: as embedRowsQ8_0Kernel in kernel.cu.
kernel void embed_rows_q8_0(device const char*   q8     [[buffer(0)]],
                            device const ushort* q8d    [[buffer(1)]],
                            device const int*    tokens [[buffer(2)]],
                            device float*        H      [[buffer(3)]],
                            constant uint&       d      [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
}

// TODO 2: as embedRowsQ4_0Kernel in kernel.cu.
kernel void embed_rows_q4_0(device const block_q4_0* q4     [[buffer(0)]],
                            device const int*        tokens [[buffer(1)]],
                            device float*            H      [[buffer(2)]],
                            constant uint&           d      [[buffer(3)]],
                            uint i [[thread_position_in_grid]]) {
}
