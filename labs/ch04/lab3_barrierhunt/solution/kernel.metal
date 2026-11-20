// Lab 4.3 reference solution (Metal): the three defects in the starter, fixed.
// Reverses each threadgroup's slice of a[0..n) in place. See kernel.cu for the
// reasoning; only the vocabulary differs.
//
// CUDA                         Metal
// threadIdx.x                  [[thread_position_in_threadgroup]]
// blockIdx.x                   [[threadgroup_position_in_grid]]
// blockDim.x                   [[threads_per_threadgroup]]
// __syncthreads()              threadgroup_barrier(mem_flags::...)
#include <metal_stdlib>
using namespace metal;

kernel void reverse_blocks(device float* a [[buffer(0)]],
                           constant int& n [[buffer(1)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint bid  [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    int t = int(tid);
    int base = int(bid * bdim);             // where this threadgroup's slice starts

    // FIX 2: a slice is one threadgroup long, whatever size the dispatch chose.
    // The starter mirrored around a literal 32.
    int size = int(bdim);

    // FIX 3: the last slice ends where the array ends. The starter mirrored it
    // around the full threadgroup size.
    int valid = min(size, n - base);        // n > base for every threadgroup dispatched

    // Phase 1: every thread reads its own element into a register.
    float mine = 0.0f;
    if (t < valid) mine = a[base + t];

    // FIX 1: no thread may write before every thread of the threadgroup has read.
    // The array is device memory, so the flag is mem_device: it "ensures the GPU
    // correctly orders the memory operations to device memory for threads in the
    // threadgroup" (MSL specification, section 6.10.1). Chapter 5's kernels pass
    // mem_threadgroup because their phases meet in threadgroup memory.
    //
    // Every thread reaches this line, including the ones past the end of the
    // array. The specification's rule is that all threads of a threadgroup
    // executing the kernel execute the barrier before any continues past it; it
    // excuses a thread that has ended on Apple silicon only.
    threadgroup_barrier(mem_flags::mem_device);

    // Phase 2: every thread writes its register to the mirrored position.
    if (t < valid) a[base + valid - 1 - t] = mine;
}
