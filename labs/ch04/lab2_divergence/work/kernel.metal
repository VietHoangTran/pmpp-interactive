// Lab 4.2 starter (Metal Shading Language).
// The three kernels of kernel.cu, with the same contract: thread i owns element
// i, the dispatch is whole threadgroups, so there can be more than n threads.
//
// paths.metal.h (beside this file, provided) gives you
//     float pathA(float x, uint iters);   float stepA(float x);
//     float pathB(float x, uint iters);   float stepB(float x);
//
// [[thread_position_in_grid]] is the global index: Metal's blockIdx * blockDim
// + threadIdx, handed to you.
#include <metal_stdlib>
using namespace metal;
#include "paths.metal.h"

// out[i] is pathA(in[i], iters) where which[i] == 0 and pathB(in[i], iters)
// everywhere else. Each thread runs ONE of the two.
kernel void branch_kernel(device const float* in    [[buffer(0)]],
                          device const int*   which [[buffer(1)]],
                          device float*       out   [[buffer(2)]],
                          constant uint&      n     [[buffer(3)]],
                          constant uint&      iters [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    // TODO 1
}

// The same out[i], written so that no thread's control flow depends on which[i].
kernel void select_kernel(device const float* in    [[buffer(0)]],
                          device const int*   which [[buffer(1)]],
                          device float*       out   [[buffer(2)]],
                          constant uint&      n     [[buffer(3)]],
                          constant uint&      iters [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    // TODO 2
}

// out[i] is in[i] after trips[i] steps of path A (stepA). trips[i] can be 0.
kernel void loop_kernel(device const float* in    [[buffer(0)]],
                        device const int*   trips [[buffer(1)]],
                        device float*       out   [[buffer(2)]],
                        constant uint&      n     [[buffer(3)]],
                        uint i [[thread_position_in_grid]]) {
    // TODO 3
}
