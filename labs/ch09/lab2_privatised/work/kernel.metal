// Lab 9.2 starter (Metal Shading Language): the same two kernels as kernel.cu.
// The host side (launch.mm) is written for you: 256 threads per threadgroup, or fewer if
// the pipeline cannot run 256, so write the kernels for any threadgroup size.
//
// CUDA                                   Metal
// __shared__ unsigned int a[N]           threadgroup atomic_uint a[N]  (in the kernel body)
// a[k] = 0 on that array                 atomic_store_explicit(&a[k], 0u, memory_order_relaxed)
// cuda::atomic_ref<unsigned int, S> r(x) atomic functions take atomic types only, so the
// r.fetch_add(v, memory_order_relaxed)   counter itself is atomic_uint and you call
//                                        atomic_fetch_add_explicit(&x, v, memory_order_relaxed).
//                                        There is no scope: the address space says who shares it.
// a plain read of an atomic_uint         atomic_load_explicit(&a[k], memory_order_relaxed)
// __syncthreads()                        threadgroup_barrier(mem_flags::mem_threadgroup)
// threadIdx.x, blockIdx.x, blockDim.x    tid, bid, bdim below
#include <metal_stdlib>
using namespace metal;

#define NUM_BINS 256      // one bin per value of an 8-bit pixel

kernel void histogram_shared(device const uchar*  image        [[buffer(0)]],
                             device atomic_uint*  bins         [[buffer(1)]],
                             constant uint&       width        [[buffer(2)]],
                             constant uint&       height       [[buffer(3)]],
                             constant uint&       coarseFactor [[buffer(4)]],
                             uint tid  [[thread_position_in_threadgroup]],
                             uint bid  [[threadgroup_position_in_grid]],
                             uint bdim [[threads_per_threadgroup]]) {
    // TODO: the five steps of histogramShared in kernel.cu, in Metal's vocabulary.
}

kernel void histogram_runs(device const uchar*  image        [[buffer(0)]],
                           device atomic_uint*  bins         [[buffer(1)]],
                           constant uint&       width        [[buffer(2)]],
                           constant uint&       height       [[buffer(3)]],
                           constant uint&       coarseFactor [[buffer(4)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint bid  [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    // TODO: the steps of histogramRuns in kernel.cu, in Metal's vocabulary.
}
