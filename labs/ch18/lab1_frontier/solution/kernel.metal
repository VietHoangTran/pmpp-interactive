// Lab 18.1 reference solution (Metal Shading Language): part A, the same search as
// visitVertexAtomically and bfsPrivatisedKernel in kernel.cu (book Figs. 18.14 and 18.15,
// pp. 441-443), one dispatch per level. The harness defines BLOCK_DIM and
// PRIVATE_FRONTIER_CAPACITY when it compiles this file.
//
// Part B (Fig. 18.17) has no Metal form: Metal documents no grid-wide barrier, no cooperative
// launch and no occupancy query from which to size a resident grid (notes/today/ch18-facts.md
// C1; ch04-facts C3, C5).
//
// What changes in Metal:
//   - Atomic functions take atomic types only, so level[] is `device atomic_uint*`, the public
//     counter is `device atomic_uint*`, and the private counter is a `threadgroup atomic_uint`
//     (notes/today/infra-facts.md C1). A read-modify-write takes no scope: the address space
//     says who shares the counter.
//   - MSL has only the weak compare-and-exchange, which may fail although the values were equal
//     (ch18-facts C4). visit_vertex therefore loops while the failure left UNVISITED in
//     `expected` (a spurious failure) and gives up when it loaded another thread's level.
//   - A threadgroup atomic_uint cannot be assigned, so the reset and the overflow store are
//     atomic_store_explicit: the data race of the page's line 29 cannot be written here (C4).
//   - The sentinel is numeric_limits<uint>::max(), the same value as UINT_MAX (C5).
//   - 256 private slots are 1 KB of threadgroup memory.
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 128
#endif
#ifndef PRIVATE_FRONTIER_CAPACITY
#define PRIVATE_FRONTIER_CAPACITY 256
#endif

constant uint UNVISITED = numeric_limits<uint>::max();

// region: visit
// Fig. 18.14. True for exactly one of the threads that try to visit the same vertex.
bool visit_vertex(device atomic_uint* level, uint vertex, uint currLevel) {
    uint expected = UNVISITED;   // fresh on every call: a real failure writes the winner's level here
    while (!atomic_compare_exchange_weak_explicit(&level[vertex], &expected, currLevel,
                                                  memory_order_relaxed, memory_order_relaxed)) {
        if (expected != UNVISITED) {
            return false;        // another thread visited it first
        }
        // expected is still UNVISITED: the failure was spurious, so try again
    }
    return true;
}
// endregion

// region: privatised
kernel void bfs_privatised(device const uint*  srcPtrs         [[buffer(0)]],
                           device const uint*  dst             [[buffer(1)]],
                           device atomic_uint* level           [[buffer(2)]],
                           device const uint*  prevFrontier    [[buffer(3)]],
                           device uint*        currFrontier    [[buffer(4)]],
                           constant uint&      numPrevFrontier [[buffer(5)]],
                           device atomic_uint* numCurrFrontier [[buffer(6)]],
                           constant uint&      currLevel       [[buffer(7)]],
                           uint i      [[thread_position_in_grid]],
                           uint tid    [[thread_position_in_threadgroup]],
                           uint tgSize [[threads_per_threadgroup]]) {
    // Initialize privatized frontier
    threadgroup uint currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    threadgroup atomic_uint numCurrFrontier_s;
    threadgroup uint currFrontierStartIdx;
    if (tid == 0) {
        atomic_store_explicit(&numCurrFrontier_s, 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Perform BFS
    if (i < numPrevFrontier) {
        uint vertex = prevFrontier[i];
        for (uint edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; ++edge) {
            uint neighbor = dst[edge];
            if (visit_vertex(level, neighbor, currLevel)) {
                uint currFrontierIdx_s = atomic_fetch_add_explicit(&numCurrFrontier_s, 1u, memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    // Overflow: put the counter back to the capacity and append to the public frontier.
                    atomic_store_explicit(&numCurrFrontier_s, (uint)PRIVATE_FRONTIER_CAPACITY, memory_order_relaxed);
                    uint currFrontierIdx = atomic_fetch_add_explicit(numCurrFrontier, 1u, memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Allocate in global frontier: one device atomic per threadgroup
    if (tid == 0) {
        uint count = atomic_load_explicit(&numCurrFrontier_s, memory_order_relaxed);
        currFrontierStartIdx = atomic_fetch_add_explicit(numCurrFrontier, count, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Commit to global frontier: consecutive threads write consecutive slots
    uint count = atomic_load_explicit(&numCurrFrontier_s, memory_order_relaxed);
    for (uint j = tid; j < count; j += tgSize) {
        currFrontier[currFrontierStartIdx + j] = currFrontier_s[j];
    }
}
// endregion
