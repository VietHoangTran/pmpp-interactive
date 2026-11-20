// Lab 3.4 reference solution (Metal Shading Language).
// Book exercises 1a and 1b (p. 64) on Metal. Square Width x Width matrices, row-major.
//
// You give [[thread_position_in_grid]] the type that fits the grid: a uint2 for the
// 2D grid of the provided kernel, a plain uint for the 1D grids of the other two.
#include <metal_stdlib>
using namespace metal;

// ---------- provided: one thread per element (the kernel of book Fig. 3.11) ----------
kernel void element_matmul(device const float* M [[buffer(0)]],
                           device const float* N [[buffer(1)]],
                           device float*       P [[buffer(2)]],
                           constant uint&  Width [[buffer(3)]],
                           uint2 gid [[thread_position_in_grid]]) {
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

// ---------- exercise 1a: one thread per output row ----------
kernel void row_matmul(device const float* M [[buffer(0)]],
                       device const float* N [[buffer(1)]],
                       device float*       P [[buffer(2)]],
                       constant uint&  Width [[buffer(3)]],
                       uint row [[thread_position_in_grid]]) {
    // launch.mm rounds the grid up to whole threadgroups, so the guard is needed.
    if (row < Width) {
        for (uint col = 0; col < Width; ++col) {
            float Pvalue = 0;
            for (uint k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}

// ---------- exercise 1b: one thread per output column ----------
kernel void col_matmul(device const float* M [[buffer(0)]],
                       device const float* N [[buffer(1)]],
                       device float*       P [[buffer(2)]],
                       constant uint&  Width [[buffer(3)]],
                       uint col [[thread_position_in_grid]]) {
    if (col < Width) {
        for (uint row = 0; row < Width; ++row) {
            float Pvalue = 0;
            for (uint k = 0; k < Width; ++k) {
                Pvalue += M[row * Width + k] * N[k * Width + col];
            }
            P[row * Width + col] = Pvalue;
        }
    }
}
