// Lab 9.1 reference solution (Metal): the starter's three defects, fixed.
// The same two kernels as kernel.cu (book Figs. 9.6 and 9.9, pp. 207 and 212).
//
// Metal has no atomic view of a plain array, as cuda::atomic_ref is. Atomic functions
// take only atomic types, so a buffer that threads update together is declared
// `device atomic_uint*` while the host still allocates plain 32-bit counters
// (notes/today/ch09-facts.md C1). A read-modify-write takes no scope either: the address
// space (device or threadgroup) is all there is. So the book's block-versus-device scope
// choice has no spelling here.
#include <metal_stdlib>
using namespace metal;

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

// D1. The starter declared bins `device uint*` and did `++bins[b];`, a plain
// read-modify-write that loses counts when threads collide. The parameter is now atomic.
// region: basic
kernel void histogram_basic(device const uchar*  image [[buffer(0)]],
                            device atomic_uint*  bins  [[buffer(1)]],
                            constant uint&       width [[buffer(2)]],
                            constant uint&       height [[buffer(3)]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint bid  [[threadgroup_position_in_grid]],
                            uint bdim [[threads_per_threadgroup]]) {
    uint i = bid*bdim + tid;
    if (i < width*height) {
        uchar b = image[i];
        atomic_fetch_add_explicit(&bins[b], 1u, memory_order_relaxed);
    }
}
// endregion

// D2. The starter had no barrier between the update and the commit. The private copy
// lives in device memory, so the barrier must order device-memory operations:
// mem_flags::mem_device ("ensures the GPU correctly orders the memory operations to
// device memory for threads in the threadgroup", MSL 6.10.1, notes/today/ch05-facts.md
// C2). mem_threadgroup would order threadgroup memory, which this kernel does not use.
// D3. The starter declared bins `device uint*` and committed with `bins[b] += binValue;`.
// Threadgroups commit to the same public bins at the same time; the add must be atomic.
// region: pool
kernel void histogram_pool(device const uchar*  image    [[buffer(0)]],
                           device atomic_uint*  bins     [[buffer(1)]],
                           device atomic_uint*  binsPool [[buffer(2)]],
                           constant uint&       width    [[buffer(3)]],
                           constant uint&       height   [[buffer(4)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint bid  [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    // This threadgroup's private histogram: NUM_BINS counters of the pool.
    device atomic_uint* binsPriv = &binsPool[bid*NUM_BINS];

    // Update the private histogram.
    uint i = bid*bdim + tid;
    if (i < width*height) {
        uchar b = image[i];
        atomic_fetch_add_explicit(&binsPriv[b], 1u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_device);

    // Commit the non-zero private bins to the public histogram. NUM_BINS may exceed the
    // threadgroup size, so each thread takes every bdim-th bin.
    for (uint b = tid; b < NUM_BINS; b += bdim) {
        uint binValue = atomic_load_explicit(&binsPriv[b], memory_order_relaxed);
        if (binValue > 0) {
            atomic_fetch_add_explicit(&bins[b], binValue, memory_order_relaxed);
        }
    }
}
// endregion
