// Lab 9.3 starter (Metal Shading Language).
// Same contract as kernel.cu; read its header first.
//
// CUDA                                   Metal
// cuda::atomic_ref<unsigned int, S> r(x)  none: declare the buffer `device atomic_uint*`
// r.fetch_add(v, memory_order_relaxed)   atomic_fetch_add_explicit(&x, v, memory_order_relaxed)
// thread scope (block, device, ...)      none on a read-modify-write: the address space decides
// blockIdx / blockDim / gridDim          [[threadgroup_position_in_grid]] / [[threads_per_threadgroup]]
//                                        / [[threadgroups_per_grid]]
// The buffer indices below must match launch.mm.
#include <metal_stdlib>
using namespace metal;

kernel void count_tokens(device const int*   tokens [[buffer(0)]],
                         device atomic_uint* counts [[buffer(1)]],
                         constant uint&      len    [[buffer(2)]],
                         constant uint&      vocab  [[buffer(3)]],
                         uint2 tid  [[thread_position_in_threadgroup]],
                         uint2 bid  [[threadgroup_position_in_grid]],
                         uint2 bdim [[threads_per_threadgroup]],
                         uint2 gdim [[threadgroups_per_grid]]) {
    // TODO: TODOs 1 to 3 of kernel.cu. launch.mm decides which grid dimension is the row.
}

kernel void append_tokens(device const int* newTokens [[buffer(0)]],
                          device uint*      counts    [[buffer(1)]],
                          constant uint&    batch     [[buffer(2)]],
                          constant uint&    vocab     [[buffer(3)]],
                          uint row [[thread_position_in_grid]]) {
    // TODO: TODO 5 of kernel.cu. Note the type of `counts` here.
}

kernel void apply_penalties(device float*      logits           [[buffer(0)]],
                            device const uint* counts           [[buffer(1)]],
                            constant uint&     n                [[buffer(2)]],
                            constant float&    frequencyPenalty [[buffer(3)]],
                            constant float&    presencePenalty  [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    // TODO: TODO 7 of kernel.cu.
}
