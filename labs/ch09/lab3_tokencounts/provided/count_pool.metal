// Lab 9.3, provided (do not edit): the Metal version of provided/count_pool.cu, Fig. 9.9's
// design applied to token counts. Read that file's header for what it does.
//
// Metal differences (notes/today/ch09-facts.md C1; ch05-facts C2):
//   - The pool is declared `device atomic_uint*`: Metal has no atomic view of a plain
//     uint. The book's plain read of a private bin in the commit becomes
//     atomic_load_explicit, and zeroing is a kernel of its own (zero_pool) dispatched
//     first in the same serial compute pass.
//   - No scope on the update: the private copy is in device memory, so this is a device
//     atomic like any other. The book's block scope has no Metal spelling.
//   - The private copies live in device memory, so the barrier between the two phases
//     orders device memory: threadgroup_barrier(mem_flags::mem_device).
#include <metal_stdlib>
using namespace metal;

kernel void zero_pool(device uint*   pool [[buffer(0)]],
                      constant uint& n    [[buffer(1)]],
                      uint i [[thread_position_in_grid]]) {
    if (i < n) pool[i] = 0;
}

kernel void count_pool(device const int*   tokens    [[buffer(0)]],
                       device atomic_uint* counts    [[buffer(1)]],
                       device atomic_uint* bins_pool [[buffer(2)]],
                       constant uint&      len       [[buffer(3)]],
                       constant uint&      vocab     [[buffer(4)]],
                       uint2 tid  [[thread_position_in_threadgroup]],
                       uint2 bid  [[threadgroup_position_in_grid]],
                       uint2 bdim [[threads_per_threadgroup]],
                       uint2 gdim [[threadgroups_per_grid]]) {
    const uint row = bid.y;
    device atomic_uint* bins_priv = bins_pool + (row * gdim.x + bid.x) * vocab;

    const uint i = bid.x * bdim.x + tid.x;
    if (i < len) {
        atomic_fetch_add_explicit(&bins_priv[tokens[row * len + i]], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    device atomic_uint* rowCounts = counts + row * vocab;
    for (uint bin = tid.x; bin < vocab; bin += bdim.x) {
        const uint binValue = atomic_load_explicit(&bins_priv[bin], memory_order_relaxed);
        if (binValue > 0) {
            atomic_fetch_add_explicit(&rowCounts[bin], binValue, memory_order_relaxed);
        }
    }
}
