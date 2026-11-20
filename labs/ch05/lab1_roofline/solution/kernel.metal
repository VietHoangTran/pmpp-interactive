// Lab 5.1 reference solution (Metal Shading Language).
#include <metal_stdlib>
using namespace metal;

// ---------- provided probe 1: memory bandwidth ----------
kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    constant uint&      n [[buffer(3)]],
                    uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < n) C[gid.x] = A[gid.x] + B[gid.x];
}

// ---------- provided probe 2: arithmetic throughput ----------
struct FmaParams { uint n; uint iters; float a; float b; };

kernel void fma_chain(device const float* in  [[buffer(0)]],
                      device float*       out [[buffer(1)]],
                      constant FmaParams&   p [[buffer(2)]],
                      uint3 gid [[thread_position_in_grid]]) {
    if (gid.x < p.n) {
        float x = in[gid.x];
        for (uint it = 0; it < p.iters; ++it) x = x * p.a + p.b;
        out[gid.x] = x;
    }
}

// ---------- your part: the naive matrix multiplication ----------
kernel void matmul(device const float* M [[buffer(0)]],
                   device const float* N [[buffer(1)]],
                   device float*       P [[buffer(2)]],
                   constant uint&  Width [[buffer(3)]],
                   uint3 gid [[thread_position_in_grid]]) {
    uint row = gid.y, col = gid.x;      // thread_position_in_grid = blockIdx*blockDim + threadIdx
    if (row < Width && col < Width) {
        float Pvalue = 0;
        for (uint k = 0; k < Width; ++k) {
            Pvalue += M[row * Width + k] * N[k * Width + col];
        }
        P[row * Width + col] = Pvalue;
    }
}
