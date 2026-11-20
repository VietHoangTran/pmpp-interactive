// Lab 9.3 reference solution (Metal Shading Language).
// Same contract and design as kernel.cu; read its header first.
//
// CUDA                                   Metal
// cuda::atomic_ref<unsigned int, S> r(x)  none: declare the buffer `device atomic_uint*`
// r.fetch_add(v, memory_order_relaxed)   atomic_fetch_add_explicit(&x, v, memory_order_relaxed)
// thread scope (block, device, ...)      none on a read-modify-write: the address space decides
// blockIdx / blockDim / gridDim          [[threadgroup_position_in_grid]] / [[threads_per_threadgroup]]
//                                        / [[threadgroups_per_grid]]
// (notes/today/ch09-facts.md C1; infra-facts C1; ch05-facts C4)
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
    const uint row = bid.y;                               // one request per grid row
    device const int* rowTokens = tokens + row * len;
    device atomic_uint* rowCounts = counts + row * vocab; // this request's histogram

    // Interleaved over all threadgroups of this row.
    for (uint i = bid.x * bdim.x + tid.x; i < len; i += gdim.x * bdim.x) {
        atomic_fetch_add_explicit(&rowCounts[rowTokens[i]], 1u, memory_order_relaxed);
    }
}

kernel void append_tokens(device const int* newTokens [[buffer(0)]],
                          device uint*      counts    [[buffer(1)]],
                          constant uint&    batch     [[buffer(2)]],
                          constant uint&    vocab     [[buffer(3)]],
                          uint row [[thread_position_in_grid]]) {
    if (row < batch) {
        // The only thread touching this row: a plain increment, on a plain uint buffer.
        counts[row * vocab + newTokens[row]] += 1;
    }
}

kernel void apply_penalties(device float*      logits           [[buffer(0)]],
                            device const uint* counts           [[buffer(1)]],
                            constant uint&     n                [[buffer(2)]],
                            constant float&    frequencyPenalty [[buffer(3)]],
                            constant float&    presencePenalty  [[buffer(4)]],
                            uint i [[thread_position_in_grid]]) {
    if (i < n) {
        const uint c = counts[i];
        logits[i] -= float(c) * frequencyPenalty + float(c > 0) * presencePenalty;
    }
}
