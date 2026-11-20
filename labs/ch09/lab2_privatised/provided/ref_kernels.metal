// Lab 9.2, provided (you do not need to edit this file): the Metal versions of the
// benchmark's two comparison kernels in ref_kernels.cu (book Figs. 9.6 and 9.12).
// The coarsening factor is a run-time argument, where the book uses a compile-time macro.
//
// Metal atomic functions take atomic types only, so the public bins and the private
// threadgroup bins are declared atomic_uint; zeroing is atomic_store_explicit and the
// commit reads with atomic_load_explicit (notes/today/ch09-facts.md C1). The threadgroup
// atomic array declared in the kernel body is the form shipping code uses (PyTorch's radix
// sort); the MSL specification shows no example of it (C1).
#include <metal_stdlib>
using namespace metal;

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

kernel void ref_histogram_basic(device const uchar*  image  [[buffer(0)]],
                                device atomic_uint*  bins   [[buffer(1)]],
                                constant uint&       width  [[buffer(2)]],
                                constant uint&       height [[buffer(3)]],
                                uint tid  [[thread_position_in_threadgroup]],
                                uint bid  [[threadgroup_position_in_grid]],
                                uint bdim [[threads_per_threadgroup]]) {
    uint i = bid*bdim + tid;
    if (i < width*height) {
        atomic_fetch_add_explicit(&bins[image[i]], 1u, memory_order_relaxed);
    }
}

// region: contiguous
kernel void ref_histogram_contiguous(device const uchar*  image        [[buffer(0)]],
                                     device atomic_uint*  bins         [[buffer(1)]],
                                     constant uint&       width        [[buffer(2)]],
                                     constant uint&       height       [[buffer(3)]],
                                     constant uint&       coarseFactor [[buffer(4)]],
                                     uint tid  [[thread_position_in_threadgroup]],
                                     uint bid  [[threadgroup_position_in_grid]],
                                     uint bdim [[threads_per_threadgroup]]) {
    // Initialize the private bins
    threadgroup atomic_uint bins_s[NUM_BINS];
    for (uint b = tid; b < NUM_BINS; b += bdim) {
        atomic_store_explicit(&bins_s[b], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Histogram: thread t takes the coarseFactor consecutive pixels starting at
    // segment + t*coarseFactor
    uint segment = coarseFactor*bid*bdim;
    for (uint c = 0; c < coarseFactor; ++c) {
        uint i = segment + tid*coarseFactor + c;
        if (i < width*height) {
            atomic_fetch_add_explicit(&bins_s[image[i]], 1u, memory_order_relaxed);
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Commit to device memory
    for (uint b = tid; b < NUM_BINS; b += bdim) {
        uint binValue = atomic_load_explicit(&bins_s[b], memory_order_relaxed);
        if (binValue > 0) {
            atomic_fetch_add_explicit(&bins[b], binValue, memory_order_relaxed);
        }
    }
}
// endregion
