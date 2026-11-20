// Lab 14.1 starter (Metal Shading Language).
// The same three kernels as kernel.cu; launch.mm drives them.
//
// CUDA                                   Metal
// blockIdx.x * blockDim.x + threadIdx.x  [[thread_position_in_grid]]
// unsigned int* hasChanged + atomicExch  device atomic_uint* + atomic_store_explicit(p, v,
//                                        memory_order_relaxed) (an atomic function needs an
//                                        atomic-typed pointer: notes/today/infra-facts.md C1)
#include <metal_stdlib>
using namespace metal;

// ---------------------------------------------------------------- part A: odd-even sort

kernel void odd_even_step(device uint*        data       [[buffer(0)]],
                          device atomic_uint* hasChanged [[buffer(1)]],
                          constant uint&      n          [[buffer(2)]],
                          constant uint&      isOddStep  [[buffer(3)]],
                          uint t [[thread_position_in_grid]]) {
    // TODO: the same three steps as oddEvenStep in kernel.cu.
}

// ---------------------------------------------------------------- part B: one-bit radix sort

kernel void radix_extract_bits(device const uint* keys [[buffer(0)]],
                               device uint*       bits [[buffer(1)]],
                               constant uint&     n    [[buffer(2)]],
                               constant uint&     bit  [[buffer(3)]],
                               uint i [[thread_position_in_grid]]) {
    // TODO: as radixExtractBits.
}

kernel void radix_scatter(device const uint* keysIn     [[buffer(0)]],
                          device const uint* valsIn     [[buffer(1)]],
                          device uint*       keysOut    [[buffer(2)]],
                          device uint*       valsOut    [[buffer(3)]],
                          device const uint* onesBefore [[buffer(4)]],
                          constant uint&     n          [[buffer(5)]],
                          constant uint&     bit        [[buffer(6)]],
                          uint i [[thread_position_in_grid]]) {
    // TODO: as radixScatter.
}
