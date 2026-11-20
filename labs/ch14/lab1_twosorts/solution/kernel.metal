// Lab 14.1 reference solution (Metal Shading Language).
// The same three kernels as kernel.cu; launch.mm drives them.
//
// CUDA                                   Metal
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// unsigned int* hasChanged + atomicExch  device atomic_uint* + atomic_store_explicit
//                                        (an atomic function needs an atomic-typed pointer:
//                                        notes/today/infra-facts.md C1)
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- part A: odd-even sort

kernel void odd_even_step(device uint*        data       [[buffer(0)]],
                          device atomic_uint* hasChanged [[buffer(1)]],
                          constant uint&      n          [[buffer(2)]],
                          constant uint&      isOddStep  [[buffer(3)]],
                          uint t [[thread_position_in_grid]]) {
    if (n < 2) {
        return;
    }
    const uint i = 2 * t + (isOddStep ? 1u : 0u);
    if (i + 1 >= n) {
        return;   // this thread's pair runs past the end (never `i < n - 1`: n - 1 wraps at 0)
    }
    const uint a = data[i];
    const uint b = data[i + 1];
    if (a > b) {
        data[i] = b;
        data[i + 1] = a;
        atomic_store_explicit(hasChanged, 1u, memory_order_relaxed);
    }
}

// ---------------------------------------------------------------- part B: one-bit radix sort

kernel void radix_extract_bits(device const uint* keys [[buffer(0)]],
                               device uint*       bits [[buffer(1)]],
                               constant uint&     n    [[buffer(2)]],
                               constant uint&     bit  [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    if (i < n) {
        bits[i] = (keys[i] >> bit) & 1u;
    } else if (i == n) {
        bits[n] = 0u;   // the scan covers n + 1 entries; this one becomes the total
    }
}

kernel void radix_scatter(device const uint* keysIn     [[buffer(0)]],
                          device const uint* valsIn     [[buffer(1)]],
                          device uint*       keysOut    [[buffer(2)]],
                          device uint*       valsOut    [[buffer(3)]],
                          device const uint* onesBefore [[buffer(4)]],
                          constant uint&     n          [[buffer(5)]],
                          constant uint&     bit        [[buffer(6)]],
                          uint i [[thread_position_in_grid]]) {
    if (i < n) {
        const uint key = keysIn[i];
        const uint b = (key >> bit) & 1u;
        const uint numOnesBefore = onesBefore[i];
        const uint numOnesTotal = onesBefore[n];
        const uint dst = b == 0 ? i - numOnesBefore : n - numOnesTotal + numOnesBefore;
        keysOut[dst] = key;
        valsOut[dst] = valsIn[i];
    }
}
