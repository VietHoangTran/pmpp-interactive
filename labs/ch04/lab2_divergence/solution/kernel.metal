// Lab 4.2 reference solution (Metal Shading Language).
// The three kernels of kernel.cu. Apple documents neither how the threads of a
// SIMD-group that take different paths are executed nor what that costs
// (notes/today/ch04-facts.md, C4), so on Metal the benchmark is the evidence.
#include <metal_stdlib>
using namespace metal;
#include "paths.metal.h"

// Fig. 4.9's shape: which[i] == 0 sends thread i down path A, anything else
// down path B.
kernel void branch_kernel(device const float* in    [[buffer(0)]],
                          device const int*   which [[buffer(1)]],
                          device float*       out   [[buffer(2)]],
                          constant uint&      n     [[buffer(3)]],
                          constant uint&      iters [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) {
        if (which[i] == 0) {
            out[i] = pathA(in[i], iters);
        } else {
            out[i] = pathB(in[i], iters);
        }
    }
}

// Both paths in every thread, then keep one value.
kernel void select_kernel(device const float* in    [[buffer(0)]],
                          device const int*   which [[buffer(1)]],
                          device float*       out   [[buffer(2)]],
                          constant uint&      n     [[buffer(3)]],
                          constant uint&      iters [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float a = pathA(in[i], iters);
        float b = pathB(in[i], iters);
        out[i] = (which[i] == 0) ? a : b;
    }
}

// Fig. 4.10's shape: a per-thread trip count.
kernel void loop_kernel(device const float* in    [[buffer(0)]],
                        device const int*   trips [[buffer(1)]],
                        device float*       out   [[buffer(2)]],
                        constant uint&      n     [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    if (i < n) {
        float x = in[i];
        int N = trips[i];
        for (int k = 0; k < N; ++k) {
            x = stepA(x);
        }
        out[i] = x;
    }
}
