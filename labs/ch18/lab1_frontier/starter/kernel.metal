// Lab 18.1 starter (Metal Shading Language): part A, the same search as visitVertexAtomically
// and bfsPrivatisedKernel in kernel.cu (book Figs. 18.14 and 18.15, pp. 441-443), one dispatch
// per level. launch.mm (written for you) runs the levels. The harness defines BLOCK_DIM and
// PRIVATE_FRONTIER_CAPACITY when it compiles this file.
//
// Part B (Fig. 18.17) has no Metal form: Metal documents no grid-wide barrier, no cooperative
// launch and no occupancy query from which to size a resident grid (notes/today/ch18-facts.md
// C1; ch04-facts C3, C5).
//
// What changes in Metal:
//   - Atomic functions take atomic types only (notes/today/infra-facts.md C1). level[] and the
//     public counter are `device atomic_uint*`; the private counter is a `threadgroup
//     atomic_uint`, read with atomic_load_explicit and written with atomic_store_explicit.
//     A read-modify-write takes no scope: the address space says who shares the counter. Use
//     memory_order_relaxed throughout.
//   - MSL has only the weak compare-and-exchange, atomic_compare_exchange_weak_explicit, which
//     may fail although the values were equal (ch18-facts C4). After a real failure `expected`
//     holds what is in memory; after a spurious one it still holds what you passed.
//   - Threadgroup arrays and variables are declared inside the kernel with `threadgroup`; the
//     barrier is threadgroup_barrier(mem_flags::mem_threadgroup).
//   - The sentinel is numeric_limits<uint>::max(), the same value as UINT_MAX (C5).
//   - `vertex` is a reserved word in MSL (it marks a vertex shader): call the vertex vertexId.
#include <metal_stdlib>
using namespace metal;

#ifndef BLOCK_DIM
#define BLOCK_DIM 128
#endif
#ifndef PRIVATE_FRONTIER_CAPACITY
#define PRIVATE_FRONTIER_CAPACITY 256
#endif

constant uint UNVISITED = numeric_limits<uint>::max();

// Fig. 18.14. True for exactly one of the threads that try to visit the same vertex.
bool visit_vertex(device atomic_uint* level, uint vertexId, uint currLevel) {
    // TODO 1: one atomic step that turns UNVISITED into currLevel, with the weak
    //         compare-and-exchange. When it returns false, look at `expected`: did another
    //         thread get there first, or should you try again?
    return false;
}

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
    // TODO 2: the threadgroup's private frontier (PRIVATE_FRONTIER_CAPACITY slots, 1 KB), its
    //         counter, and a place for the start index. One thread zeroes the counter; who
    //         waits for it?

    // TODO 3: one thread per vertex of the previous frontier, as in kernel.cu: visit each
    //         neighbour, take a private slot, and on overflow put the counter back to the
    //         capacity and append to the public frontier instead.

    // TODO 4: wait until no thread can add to the private frontier.

    // TODO 5: one thread reserves the threadgroup's range of the public frontier with one
    //         atomic. Then wait.

    // TODO 6: commit with coalesced stores. The threadgroup may have fewer than BLOCK_DIM
    //         threads (launch.mm asks for fewer if the pipeline cannot run that many): stride
    //         by tgSize.
}
