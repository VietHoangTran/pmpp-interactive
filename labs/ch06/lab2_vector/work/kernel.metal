// Lab 6.2 starter (Metal Shading Language).
// Book: section 6.3 (pp. 138-139), Fig. 6.12, Exercise 4; section 6.5 (pp. 141-143).
//
// Two Metal differences from kernel.cu (notes/today/ch06-facts.md):
// - No pointer cast. Apple's documented way to read four floats at once is to declare
//   the buffer with a vector element type, `device const float4*` (C3; infra-facts C7).
//   A float4 view cannot reach the 1 to 3 elements of a partial last chunk, so vec_add4
//   also takes the same three buffers as plain floats. launch.mm binds each one twice.
// - Index with int, not uint. Apple says uint indexing can stop the compiler from
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
    // TODO: the same four steps as vecAdd4Kernel in kernel.cu. Full chunks go through
    //       the float4 views, the partial last chunk through the float views.
}

kernel void vec_add_coarse(device const float* x [[buffer(0)]],
                           device const float* y [[buffer(1)]],
                           device float*       z [[buffer(2)]],
                           constant int&       n [[buffer(3)]],
                           uint gid [[thread_position_in_grid]]) {
    // TODO: the coarsening loop of vecAddCoarseKernel. Apple documents no unroll
    //       control; `#pragma clang loop unroll(full)` before the loop is accepted by
    //       the compiler and used by MLX and llama.cpp (C4).
}
