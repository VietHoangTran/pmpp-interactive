// Section 11.9, "Make the flag safe", tab "MSL 4.1": the flag protocol of Fig. 11.17
// (pp. 278-279) with MSL 4.1's acquire and release, which exist only in the overloads of
// atomic_load_explicit and atomic_store_explicit that take mem_flags
// (notes/today/ch11-facts.md C4). partialSums is declared coherent(device), because by
// default device memory is only threadgroup-coherent (C3).
//
// Compiled, never run. app/src/chapters/ch11/S119.snippets.test.ts builds this file at
// -std=metal4.1 and checks that -std=metal4.0 rejects it. Metal documents no forward
// progress between threadgroups (C5), so a kernel that spins on another threadgroup's flag
// is not something the course runs; on Metal the global scan is lab 11.4's.
#include <metal_stdlib>
using namespace metal;

// The leader of block b - 1: a plain store of the sum, then the flag with release.
void publish(coherent(device) device float* partialSums, device atomic_uint* flags,
             uint bid, float prev, float blockSum) {
    partialSums[bid] = prev + blockSum;
    atomic_store_explicit(&flags[bid], 1u, memory_order_release, mem_flags::mem_device);
}

// The leader of block b: spin on the flag with acquire, then a plain load of the sum.
float waitFor(coherent(device) device float* partialSums, device atomic_uint* flags, uint bid) {
    while (atomic_load_explicit(&flags[bid - 1], memory_order_acquire, mem_flags::mem_device) == 0) {}
    return partialSums[bid - 1];
}

// One leader per threadgroup, its index claimed from a counter as in Fig. 11.18 (p. 280).
// The segment scans are left out: blockSums holds each segment's total.
kernel void flag_protocol_leader(coherent(device) device float* partialSums [[buffer(0)]],
                                 device atomic_uint* flags [[buffer(1)]],
                                 device const float* blockSums [[buffer(2)]],
                                 device atomic_uint* blockCounter [[buffer(3)]],
                                 uint tid [[thread_position_in_threadgroup]]) {
    if (tid != 0) return;
    uint bid = atomic_fetch_add_explicit(blockCounter, 1u, memory_order_relaxed);
    float prev = bid > 0 ? waitFor(partialSums, flags, bid) : 0.0f;
    publish(partialSums, flags, bid, prev, blockSums[bid]);
}
