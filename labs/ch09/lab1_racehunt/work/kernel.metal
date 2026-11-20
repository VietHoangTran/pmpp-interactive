// Lab 9.1 starter (Metal): the same two kernels as kernel.cu, with the same THREE
// defects. The host side, launch.mm, is correct.
//
// CUDA                                   Metal
// cuda::atomic_ref<unsigned int, S> r(x) no atomic view of a plain object: atomic functions
// r.fetch_add(v, memory_order_relaxed)   take atomic types only, so the pointer itself is
//                                        declared atomic (device atomic_uint* p) and you call
//                                        atomic_fetch_add_explicit(&p[k], v, memory_order_relaxed)
// a plain read of an atomic_uint         atomic_load_explicit(&p[k], memory_order_relaxed)
// __syncthreads()                        threadgroup_barrier(mem_flags::...): the flag names
//                                        the memory whose operations the barrier orders
// threadIdx.x, blockIdx.x, blockDim.x    tid, bid, bdim below
#include <metal_stdlib>
using namespace metal;

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

kernel void histogram_basic(device const uchar*  image [[buffer(0)]],
                            device uint*         bins  [[buffer(1)]],
                            constant uint&       width [[buffer(2)]],
                            constant uint&       height [[buffer(3)]],
                            uint tid  [[thread_position_in_threadgroup]],
                            uint bid  [[threadgroup_position_in_grid]],
                            uint bdim [[threads_per_threadgroup]]) {
    uint i = bid*bdim + tid;
    if (i < width*height) {
        uchar b = image[i];
        ++bins[b];
    }
}

kernel void histogram_pool(device const uchar*  image    [[buffer(0)]],
                           device uint*         bins     [[buffer(1)]],
                           device atomic_uint*  binsPool [[buffer(2)]],
                           constant uint&       width    [[buffer(3)]],
                           constant uint&       height   [[buffer(4)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint bid  [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    // This threadgroup's private histogram: NUM_BINS counters of the pool.
    device atomic_uint* binsPriv = &binsPool[bid*NUM_BINS];

    // Update the private histogram. Only this threadgroup's threads touch binsPriv.
    uint i = bid*bdim + tid;
    if (i < width*height) {
        uchar b = image[i];
        atomic_fetch_add_explicit(&binsPriv[b], 1u, memory_order_relaxed);
    }

    // Commit the non-zero private bins to the public histogram. NUM_BINS may exceed the
    // threadgroup size, so each thread takes every bdim-th bin.
    for (uint b = tid; b < NUM_BINS; b += bdim) {
        uint binValue = atomic_load_explicit(&binsPriv[b], memory_order_relaxed);
        if (binValue > 0) {
            bins[b] += binValue;
        }
    }
}
