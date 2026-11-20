// Lab 4.3 starter (Metal): reverses each threadgroup's slice of a[0..n) in place.
// Threadgroup b owns the elements b*bdim up to (b+1)*bdim, and the last slice
// stops where the array stops. Every thread reads its element into a register,
// then writes it to the mirrored position of its slice.
//
// The same THREE defects as in kernel.cu. Fix them there first, on the emulator,
// then carry the fixes over.
//
// CUDA                         Metal
// threadIdx.x                  [[thread_position_in_threadgroup]]
// blockIdx.x                   [[threadgroup_position_in_grid]]
// blockDim.x                   [[threads_per_threadgroup]]
// __syncthreads()              threadgroup_barrier(mem_flags::...)
//   The flag names the memory whose accesses the barrier has to order:
//   mem_threadgroup for threadgroup memory, mem_device for device memory.
#include <metal_stdlib>
using namespace metal;

kernel void reverse_blocks(device float* a [[buffer(0)]],
                           constant int& n [[buffer(1)]],
                           uint tid  [[thread_position_in_threadgroup]],
                           uint bid  [[threadgroup_position_in_grid]],
                           uint bdim [[threads_per_threadgroup]]) {
    int t = int(tid);
    int base = int(bid * bdim);             // where this threadgroup's slice starts
    int i = base + t;

    float mine = 0.0f;
    if (i < n) mine = a[i];

    if (i < n) a[base + 32 - 1 - t] = mine;
}
