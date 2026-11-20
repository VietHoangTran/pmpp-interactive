// Lab 2.2 (Metal Shading Language). Provided complete: this is the kernel of
// lab 2.1. Not in the book.
#include <metal_stdlib>
using namespace metal;

kernel void vec_add(device const float* A [[buffer(0)]],
                    device const float* B [[buffer(1)]],
                    device float*       C [[buffer(2)]],
                    uint i [[thread_position_in_grid]]) {
    // No index formula and no guard: launch.mm dispatches exactly n threads,
    // and Metal hands each one its position in the grid.
    C[i] = A[i] + B[i];
}
