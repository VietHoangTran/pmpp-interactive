// Provided baseline for Metal: one thread per output element, every operand
// fetched from device memory. Same algorithm as shared/naive_matmul.cu.
#include <metal_stdlib>
using namespace metal;

kernel void naive_matmul(device const float* M [[buffer(0)]],
                         device const float* N [[buffer(1)]],
                         device float*       P [[buffer(2)]],
                         constant uint&  Width [[buffer(3)]],
                         uint3 gid [[thread_position_in_grid]]) {
    uint row = gid.y;
    uint col = gid.x;
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (uint k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}
