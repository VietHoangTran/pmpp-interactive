// Lab 9.2 reference solution (Metal): the same two kernels as kernel.cu (book Figs. 9.14
// and 9.15, pp. 217 and 218), with the coarsening factor as a run-time argument.
//
// Metal atomic functions take atomic types only (notes/today/ch09-facts.md C1). So the
// public bins are `device atomic_uint*`, the private bins are a `threadgroup atomic_uint`
// array, zeroing is atomic_store_explicit, and the commit reads each private bin with
// atomic_load_explicit, where CUDA uses plain accesses. A read-modify-write takes no scope:
// the address space (threadgroup or device) is all that says who shares the counter.
// The threadgroup atomic array declared in the kernel body is the form shipping code uses
// (PyTorch's radix sort); the MSL specification shows no example of it (C1).
#include <metal_stdlib>
using namespace metal;

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

// region: shared
kernel void histogram_shared(device const uchar*  image        [[buffer(0)]],
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

    // Histogram: in iteration c the threadgroup reads bdim consecutive pixels
    uint segment = coarseFactor*bid*bdim;
    for (uint c = 0; c < coarseFactor; ++c) {
        uint i = segment + c*bdim + tid;
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

// region: runs
kernel void histogram_runs(device const uchar*  image        [[buffer(0)]],
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

    // Histogram, one run at a time: b is the value of the current run, bin_r its length
    uint segment = coarseFactor*bid*bdim;
    if (segment + tid < width*height) {
        uint b = image[segment + tid];
        uint bin_r = 1;
        for (uint c = 1; c < coarseFactor; ++c) {
            uint i = segment + c*bdim + tid;
            if (i < width*height) {
                uchar bNext = image[i];
                if (bNext == b) {
                    ++bin_r;
                } else {
                    atomic_fetch_add_explicit(&bins_s[b], bin_r, memory_order_relaxed);
                    b = bNext;
                    bin_r = 1;
                }
            }
        }
        // The last run has not been added yet
        atomic_fetch_add_explicit(&bins_s[b], bin_r, memory_order_relaxed);
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
