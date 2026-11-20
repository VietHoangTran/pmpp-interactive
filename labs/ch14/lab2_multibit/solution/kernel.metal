// Lab 14.2 reference solution (Metal Shading Language). Ours: the book gives no code for
// sections 14.6 to 14.8. The same two kernels as kernel.cu; launch.mm drives them.
//
// CUDA                                  Metal
// extern __shared__ unsigned int s[]    threadgroup uint* s [[threadgroup(0)]], sized by launch.mm
// __syncthreads()                       threadgroup_barrier(mem_flags::mem_threadgroup)
// gridDim.x                             [[threadgroups_per_grid]]
//
// One difference from kernel.cu: the block-level scan of each one-bit split uses
// simd_prefix_exclusive_sum and simd_sum inside each SIMD-group (notes/today/infra-facts.md C2)
// and one threadgroup slot per SIMD-group for their totals. The SIMD-group width is not a
// documented constant (infra C5), so launch.mm reads it from the pipeline and sizes that array
// ([[threadgroup(1)]]); the kernel loops over [[simdgroups_per_threadgroup]] of them.
//
// A second difference follows from it. How a threadgroup is split into SIMD-groups is
// implementation-defined (MSL 5.2.3.6, notes/today/ch04-facts.md C2), and this scan adds up
// the thread totals in (SIMD-group, lane) order. So in the splits a thread owns the tile
// positions of its rank in that order, me = sg * width + lane, not of
// thread_position_in_threadgroup. The coalesced loads and stores keep t: any one-to-one
// assignment of positions to threads is right there.
#include <metal_stdlib>
using namespace metal;

#define BLOCK_DIM 256
#define MAX_KEYS_PER_THREAD 8
#define MAX_RADIX_BITS 8
#define MAX_BUCKETS 256   // 2^MAX_RADIX_BITS

// Exclusive scan of one value per thread over the threadgroup. Every thread must call it, in
// uniform control flow. Its barrier also orders every thread's earlier reads of threadgroup
// memory before any thread's later writes.
static uint threadgroup_exclusive_scan(uint value, threadgroup uint* sgTotals, uint lane, uint sg, uint numSg,
                                       thread uint& total) {
    const uint inGroup = simd_prefix_exclusive_sum(value);
    const uint groupSum = simd_sum(value);
    if (lane == 0) {
        sgTotals[sg] = groupSum;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint before = 0;
    uint all = 0;
    for (uint g = 0; g < numSg; ++g) {
        const uint s = sgTotals[g];
        before += g < sg ? s : 0u;
        all += s;
    }
    total = all;
    return before + inGroup;
}

kernel void radix_local_sort(device const uint* keysIn        [[buffer(0)]],
                             device const uint* valsIn        [[buffer(1)]],
                             device uint*       keysLocal     [[buffer(2)]],
                             device uint*       valsLocal     [[buffer(3)]],
                             device uint*       table         [[buffer(4)]],
                             device uint*       localStarts   [[buffer(5)]],
                             constant uint&     n             [[buffer(6)]],
                             constant uint&     shift         [[buffer(7)]],
                             constant uint&     radixBits     [[buffer(8)]],
                             constant uint&     keysPerThread [[buffer(9)]],
                             threadgroup uint*  tile_s        [[threadgroup(0)]],   // keys, then values: 2 * 256c
                             threadgroup uint*  sgTotals      [[threadgroup(1)]],   // one per SIMD-group
                             uint t         [[thread_position_in_threadgroup]],
                             uint block     [[threadgroup_position_in_grid]],
                             uint numBlocks [[threadgroups_per_grid]],
                             uint lane      [[thread_index_in_simdgroup]],
                             uint sg        [[simdgroup_index_in_threadgroup]],
                             uint numSg     [[simdgroups_per_threadgroup]],
                             uint width     [[threads_per_simdgroup]]) {
    threadgroup uint start_s[MAX_BUCKETS + 1];

    const uint me = sg * width + lane;   // this thread's rank in the scan's order
    const uint c = keysPerThread;
    const uint tile = BLOCK_DIM * c;
    const uint base = block * tile;
    const uint count = min(tile, n - base);
    threadgroup uint* keys_s = tile_s;
    threadgroup uint* vals_s = tile_s + tile;

    // 1. Coalesced load.
    for (uint k = 0; k < c; ++k) {
        const uint p = k * BLOCK_DIM + t;
        if (p < count) {
            keys_s[p] = keysIn[base + p];
            vals_s[p] = valsIn[base + p];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 2. r stable one-bit splits; the thread of rank me owns tile positions me*c .. me*c + c - 1,
    //    so the positions follow the order in which the scan adds up the thread totals.
    for (uint b = 0; b < radixBits; ++b) {
        const uint bit = shift + b;
        uint myKeys[MAX_KEYS_PER_THREAD];
        uint myVals[MAX_KEYS_PER_THREAD];
        uint myOnes = 0;
        for (uint k = 0; k < c; ++k) {
            const uint p = me * c + k;
            if (p < count) {
                myKeys[k] = keys_s[p];
                myVals[k] = vals_s[p];
                myOnes += (myKeys[k] >> bit) & 1u;
            }
        }
        uint onesTotal;
        uint onesBefore = threadgroup_exclusive_scan(myOnes, sgTotals, lane, sg, numSg, onesTotal);
        const uint zerosTotal = count - onesTotal;
        for (uint k = 0; k < c; ++k) {
            const uint p = me * c + k;
            if (p < count) {
                const uint one = (myKeys[k] >> bit) & 1u;
                const uint dst = one ? zerosTotal + onesBefore : p - onesBefore;
                keys_s[dst] = myKeys[k];
                vals_s[dst] = myVals[k];
                onesBefore += one;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // 3. Where each local bucket starts (see kernel.cu).
    const uint buckets = 1u << radixBits;
    const uint mask = buckets - 1;
    for (uint d = t; d <= buckets; d += BLOCK_DIM) {
        start_s[d] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint k = 0; k < c; ++k) {
        const uint p = k * BLOCK_DIM + t;
        if (p < count) {
            const uint digit = (keys_s[p] >> shift) & mask;
            const uint first = p == 0 ? 0u : ((keys_s[p - 1] >> shift) & mask) + 1;
            for (uint d = first; d <= digit; ++d) {
                start_s[d] = p;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 4. Digit-major table and local starts.
    for (uint d = t; d < buckets; d += BLOCK_DIM) {
        table[d * numBlocks + block] = start_s[d + 1] - start_s[d];
        localStarts[block * buckets + d] = start_s[d];
    }

    // 5. Coalesced store of the sorted tile.
    for (uint k = 0; k < c; ++k) {
        const uint p = k * BLOCK_DIM + t;
        if (p < count) {
            keysLocal[base + p] = keys_s[p];
            valsLocal[base + p] = vals_s[p];
        }
    }
}

kernel void radix_scatter_local(device const uint* keysLocal     [[buffer(0)]],
                                device const uint* valsLocal     [[buffer(1)]],
                                device uint*       keysOut       [[buffer(2)]],
                                device uint*       valsOut       [[buffer(3)]],
                                device const uint* tableScanned  [[buffer(4)]],
                                device const uint* localStarts   [[buffer(5)]],
                                constant uint&     n             [[buffer(6)]],
                                constant uint&     shift         [[buffer(7)]],
                                constant uint&     radixBits     [[buffer(8)]],
                                constant uint&     keysPerThread [[buffer(9)]],
                                uint t         [[thread_position_in_threadgroup]],
                                uint block     [[threadgroup_position_in_grid]],
                                uint numBlocks [[threadgroups_per_grid]]) {
    threadgroup uint offset_s[MAX_BUCKETS];

    const uint tile = BLOCK_DIM * keysPerThread;
    const uint base = block * tile;
    const uint count = min(tile, n - base);
    const uint buckets = 1u << radixBits;
    const uint mask = buckets - 1;

    for (uint d = t; d < buckets; d += BLOCK_DIM) {
        offset_s[d] = tableScanned[d * numBlocks + block] - localStarts[block * buckets + d];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint k = 0; k < keysPerThread; ++k) {
        const uint p = k * BLOCK_DIM + t;
        if (p < count) {
            const uint key = keysLocal[base + p];
            const uint dst = offset_s[(key >> shift) & mask] + p;
            keysOut[dst] = key;
            valsOut[dst] = valsLocal[base + p];
        }
    }
}
