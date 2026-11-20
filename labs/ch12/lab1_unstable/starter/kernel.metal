// Lab 12.1 starter (Metal Shading Language): the same three unstable filters as kernel.cu,
// keep = key < threshold. The caller zeroes outputSize. The harness defines BLOCK_DIM when it
// compiles this file; a threadgroup may have fewer threads than that, never more.
//
// CUDA                                   Metal
// cuda::atomic_ref<unsigned int, S> r(x) atomic functions take atomic types only, so the
// r.fetch_add(v, memory_order_relaxed)   counter is atomic_uint and you call
//                                        atomic_fetch_add_explicit(&x, v, memory_order_relaxed).
//                                        There is no scope: the address space says who shares it.
// a plain read of an atomic_uint         atomic_load_explicit(&x, memory_order_relaxed)
// x = 0 on an atomic_uint                atomic_store_explicit(&x, 0u, memory_order_relaxed)
// __shared__ T a[N]                      threadgroup T a[N]   (declared in the kernel body)
// __syncthreads()                        threadgroup_barrier(mem_flags::mem_threadgroup)
// blockIdx.x*blockDim.x + threadIdx.x    i  [[thread_position_in_grid]]
// threadIdx.x                            tid [[thread_position_in_threadgroup]]
//
// For the warp-aggregated kernel Metal has no lane masks and no cooperative groups, but one
// SIMD-group function for each step of Fig. 12.3 (notes/today/ch12-facts.md C1):
//   simd_sum(x)                  the sum of x over the SIMD-group, given to every thread
//   simd_prefix_exclusive_sum(x) the sum of x over the threads with a lower index
//   simd_is_first()              true in the first active thread of the SIMD-group
//   simd_broadcast_first(x)      the first active thread's x, given to every thread
// They work over the threads that are active when they are called. Apple does not say
// whether those are exactly the threads that took a branch (C3), so call them from every
// thread, outside any branch.
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

kernel void filter_atomic(device const uint*  input      [[buffer(0)]],
                          device uint*        output     [[buffer(1)]],
                          constant uint&      n          [[buffer(2)]],
                          constant uint&      threshold  [[buffer(3)]],
                          device atomic_uint* outputSize [[buffer(4)]],
                          uint i [[thread_position_in_grid]]) {
    // TODO: TODO 1 of kernel.cu, in Metal's vocabulary.
}

kernel void filter_warp_aggregated(device const uint*  input      [[buffer(0)]],
                                   device uint*        output     [[buffer(1)]],
                                   constant uint&      n          [[buffer(2)]],
                                   constant uint&      threshold  [[buffer(3)]],
                                   device atomic_uint* outputSize [[buffer(4)]],
                                   uint i [[thread_position_in_grid]]) {
    // TODO: every thread works out whether it keeps a key (no early return). Then, from
    //       every thread: how many threads of the SIMD-group keep, and how many keep below
    //       this one. One thread reserves the SIMD-group's range with one atomic, and only
    //       if there is something to reserve. Every thread learns the start, and a thread
    //       that keeps stores its key at start + its rank.
}

kernel void filter_privatised(device const uint*  input      [[buffer(0)]],
                              device uint*        output     [[buffer(1)]],
                              constant uint&      n          [[buffer(2)]],
                              constant uint&      threshold  [[buffer(3)]],
                              device atomic_uint* outputSize [[buffer(4)]],
                              uint i   [[thread_position_in_grid]],
                              uint tid [[thread_position_in_threadgroup]]) {
    // TODO: TODOs 7 to 10 of kernel.cu, in Metal's vocabulary. The private counter is a
    //       threadgroup atomic_uint; every read of it goes through atomic_load_explicit.
}
