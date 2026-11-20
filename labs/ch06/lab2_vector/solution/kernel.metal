// Lab 6.2 reference solution (Metal Shading Language).
// Book: section 6.3 (pp. 138-139), Fig. 6.12, Exercise 4; section 6.5 (pp. 141-143).
//
// Two Metal differences from kernel.cu (notes/today/ch06-facts.md):
// - No pointer cast. Apple's documented way to read four floats at once is to declare
//   the buffer with a vector element type, `device const float4*` (C3; infra-facts C7).
//   A float4 view cannot reach the 1 to 3 elements of a partial last chunk, so the host
//   binds each buffer twice: as float4 at indices 0-2 and as float at indices 3-5.
//   (Casting a float pointer to a float4 pointer compiles, and MLX and llama.cpp do it,
//   but the specification does not describe it.)
// - Indices are int, not uint. Apple says uint indexing can stop the compiler from
//   vectorising loads, because uint arithmetic must wrap (C3).
#include <metal_stdlib>
using namespace metal;

#define COARSENING_FACTOR 4

kernel void vec_add4(device const float4* x4 [[buffer(0)]],   // x, y, z seen as float4
                     device const float4* y4 [[buffer(1)]],
                     device float4*       z4 [[buffer(2)]],
                     device const float*  x  [[buffer(3)]],   // the same buffers seen as float
                     device const float*  y  [[buffer(4)]],
                     device float*        z  [[buffer(5)]],
                     constant int&        n  [[buffer(6)]],
                     uint gid [[thread_position_in_grid]]) {
    int i = int(gid);           // names the chunk 4i .. 4i+3
    int first = 4 * i;
    if (first + 3 < n) {
        z4[i] = x4[i] + y4[i];  // one float4 access per array; + adds componentwise
    } else if (first < n) {
        for (int j = first; j < n; ++j) z[j] = x[j] + y[j];   // partial last chunk
    }
    // first >= n: past the data, nothing to do.
}

kernel void vec_add_coarse(device const float* x [[buffer(0)]],
                           device const float* y [[buffer(1)]],
                           device float*       z [[buffer(2)]],
                           constant int&       n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    int iStart = COARSENING_FACTOR * int(gid);
    // Apple documents no unroll control. This Clang pragma is accepted by the compiler
    // and used by MLX and llama.cpp; whether the GPU compiler honours it is not
    // documented (C4).
    #pragma clang loop unroll(full)
    for (int c = 0; c < COARSENING_FACTOR; ++c) {
        int i = iStart + c;
        if (i < n) z[i] = x[i] + y[i];
    }
}
