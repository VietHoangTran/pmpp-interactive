// Lab 18.1 starter (CUDA; the same file is built for HIP and for the two CPU emulators).
// Breadth-first search with frontiers, book sections 18.5 to 18.7 (pp. 438-448).
//   visitVertexAtomically   Fig. 18.14 (p. 441), part A: label a vertex exactly once.
//   bfsPrivatisedKernel     Fig. 18.15 (p. 443), part A: one launch per level, with a private
//                           frontier per block in shared memory.
//   bfsCooperativeKernel    Fig. 18.17 (p. 447), part B (optional): every level in one launch,
//                           with grid-wide barriers between levels.
//
// The graph is in CSR form (section 18.1, p. 427): the out-neighbours of vertex v are
// dst[srcPtrs[v]] .. dst[srcPtrs[v + 1] - 1]. level[v] is v's BFS level, UNVISITED until found.
// The driver sets level[root] = 0 and every other level to UNVISITED before each search.
//
// Where this lab differs from the page (README.md, "Where this differs from the page"):
//   - Raw pointers and sizes (srcPtrs, dst, numVertices) instead of the book's CSRGraph.
//   - UNVISITED is UINT_MAX from <climits>.
//   - Fig. 18.15 line 29 resets the private counter with a plain store while other threads of
//     the block update it atomically. That is a data race. Write it so that it is not one.
//   - Fig. 18.17's __shared__ variables go at the top of the kernel, not inside the level loop.
//   - grid.thread_rank() and grid.num_threads() are 64-bit on CUDA and 32-bit on HIP: cast
//     them to unsigned int once.
//   - cudaLaunchCooperativeKernel takes all six arguments (HIP has no defaults for the last two).
//
// On HIP, <cuda/atomic> comes from libhipcxx, which this course's HIP build does not rely on
// (notes/today/infra-facts.md B2), and its compare_exchange_strong is lowered to a weak
// compare-and-swap (notes/today/ch18-facts.md B4). Beside each cuda::atomic_ref, write the
// legacy call for HIP, which is relaxed and works on shared and global memory (infra B1):
//     #if defined(__HIPCC__)
//         ... atomicCAS(address, compare, value) ...   (strong; returns the OLD value)
//         ... atomicAdd(address, value) ...            (returns the old value)
//         ... atomicExch(address, value) ...           (an atomic store, as far as you need)
//     #else
//         ... cuda::atomic_ref<unsigned int, scope> ... (relaxed orders) ...
//     #endif
#include <climits>

#if !defined(__HIPCC__)
#include <cuda/atomic>
#endif

// The cooperative-groups header has a different path on HIP (infra-facts B4).
#if defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
namespace cg = cooperative_groups;

#define UNVISITED UINT_MAX

// Threads per block, and slots in each block's private frontier. The capacity is larger than
// the block, so a block can find more vertices than it has threads. The build may define both.
#ifndef BLOCK_DIM
#define BLOCK_DIM 128
#endif
#ifndef PRIVATE_FRONTIER_CAPACITY
#define PRIVATE_FRONTIER_CAPACITY 256
#endif
static_assert(BLOCK_DIM % 64 == 0 && BLOCK_DIM <= 1024, "BLOCK_DIM: a multiple of 64, at most 1024");

// What launchBfsCooperative returns (the driver reads the same numbers).
#define BFS_COOP_DONE         0   // the search ran
#define BFS_COOP_NOT_WRITTEN  1   // part B not written yet: the starter returns this
#define BFS_COOP_UNSUPPORTED  2   // this device reports no cooperative launch
#define BFS_COOP_LAUNCH_ERROR 3   // cudaLaunchCooperativeKernel returned an error

// ================================================================ part A

// Fig. 18.14. Returns true for exactly one of the threads that try to visit the same vertex.
__device__ inline unsigned int visitVertexAtomically(unsigned int vertex, unsigned int* level,
                                                     unsigned int currLevel) {
    // TODO 1: "is level[vertex] still UNVISITED? then make it currLevel" must be ONE atomic
    //         step, or two threads can both see UNVISITED and both put the vertex in the
    //         frontier. Which operation is that, and at which scope (threads of other blocks
    //         visit the same vertex)?
    //         compare_exchange_strong takes its expected value by reference and overwrites it
    //         when the swap fails. What would a failed call leave behind for the next call if
    //         the variable outlived it?
    return 0;
}

__global__ void bfsPrivatisedKernel(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                    unsigned int* level, unsigned int* prevFrontier, unsigned int* currFrontier,
                                    unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
                                    unsigned int currLevel) {
    // TODO 2: the block's private frontier (PRIVATE_FRONTIER_CAPACITY slots) and its counter,
    //         in shared memory. Shared memory starts with whatever was there: one thread sets
    //         the counter to 0. Who has to wait for that before going on?

    // TODO 3: one thread per vertex of the previous frontier. Walk its out-edges; for each
    //         neighbour that this thread visits (TODO 1), take a slot in the private frontier
    //         from the private counter. Which threads share that counter, so which scope?
    //         - A slot below the capacity: store the neighbour there.
    //         - Otherwise the private frontier is full. Append the neighbour straight to the
    //           public frontier (currFrontier, counter *numCurrFrontier), and put the private
    //           counter back to the capacity, so that the steps below never read past the
    //           private array. Other threads may be incrementing that counter right now: what
    //           kind of store is that?

    // TODO 4: no thread may add to the private frontier after this point. How do you know
    //         they have all finished?

    // TODO 5: one thread reserves the block's range in the public frontier with ONE atomic on
    //         *numCurrFrontier (how many slots?) and leaves the start of the range in shared
    //         memory. What must be true before the other threads read that start?

    // TODO 6: commit: copy the private entries to the public frontier from that start. The
    //         block may hold more entries than it has threads. Arrange the loop so that
    //         neighbouring threads write neighbouring addresses (p. 444).
}

// The host loop for bfsPrivatisedKernel, one launch per level (written for you).
// Before the call: level[v] = UNVISITED for every v except level[root] = 0, and frontierA and
// frontierB each have room for numVertices entries. Returns the number of launches; the last
// one finds no new vertex. If frontierSizes is not null (a host array of numVertices + 1
// entries), frontierSizes[0] = 1 (the root) and frontierSizes[L] = the number of vertices the
// launch for level L found.
unsigned int launchBfsPrivatised(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                 unsigned int root, unsigned int* level, unsigned int* frontierA,
                                 unsigned int* frontierB, unsigned int* numCurrFrontier,
                                 unsigned int* frontierSizes) {
    unsigned int* prevFrontier = frontierA;
    unsigned int* currFrontier = frontierB;
    cudaMemcpy(prevFrontier, &root, sizeof(unsigned int), cudaMemcpyHostToDevice);
    unsigned int numPrevFrontier = 1;
    if(frontierSizes) {
        frontierSizes[0] = 1;
    }
    unsigned int launches = 0;
    for(unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        cudaMemset(numCurrFrontier, 0, sizeof(unsigned int));
        unsigned int numBlocks = (numPrevFrontier + BLOCK_DIM - 1) / BLOCK_DIM;
        bfsPrivatisedKernel<<<numBlocks, BLOCK_DIM>>>(srcPtrs, dst, numVertices, level, prevFrontier,
                                                       currFrontier, numPrevFrontier, numCurrFrontier, currLevel);
        ++launches;
        unsigned int numCurrFrontier_h = 0;
        cudaMemcpy(&numCurrFrontier_h, numCurrFrontier, sizeof(unsigned int), cudaMemcpyDeviceToHost);
        if(frontierSizes) {
            frontierSizes[currLevel] = numCurrFrontier_h;
        }
        // A correct kernel finds each vertex once, so it never fills a frontier past numVertices
        // or runs for more than numVertices levels. Stop instead of reading past the buffers.
        if(numCurrFrontier_h > numVertices || currLevel >= numVertices) {
            break;
        }
        numPrevFrontier = numCurrFrontier_h;
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
    return launches;
}

// ================================================================ part B (optional)

// Every level in one launch. Launch only through launchBfsCooperative: grid.sync() needs a
// cooperative launch, and every block of the grid must be resident at once (p. 445).
__global__ void bfsCooperativeKernel(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                     unsigned int* level, unsigned int* prevFrontier, unsigned int* currFrontier,
                                     unsigned int numPrevFrontier, unsigned int* numCurrFrontier) {
    // TODO 7: declare the shared variables here, at the top: they live as long as the block.
    //         Get the grid group, and this thread's rank and the grid's thread count as
    //         unsigned int.

    // TODO 8: loop over levels from 1 while the previous frontier is not empty. Each level is
    //         part A's kernel body, with one change: the grid is smaller than a large frontier,
    //         so each thread strides through the previous frontier by the grid's thread count.

    // TODO 9: end of a level. Every thread must learn the new frontier's size, the counter
    //         must be zeroed for the next level, and the two frontiers swap roles. The book
    //         does it with three grid.sync() calls (pp. 447-448). For each one, what goes
    //         wrong without it?
}

// The host side of part B. Same contract as launchBfsPrivatised, without the per-level sizes:
// the host sees nothing until the whole search has finished. Returns one of BFS_COOP_*.
int launchBfsCooperative(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                         unsigned int root, unsigned int* level, unsigned int* frontierA,
                         unsigned int* frontierB, unsigned int* numCurrFrontier) {
    // Written for you: a device that cannot launch cooperatively cannot run part B.
    int supportsCoopLaunch = 0;
    cudaDeviceGetAttribute(&supportsCoopLaunch, cudaDevAttrCooperativeLaunch, 0);
    if(!supportsCoopLaunch) {
        return BFS_COOP_UNSUPPORTED;
    }
    // Written for you: the root is the first frontier, and the counter starts at zero.
    cudaMemcpy(frontierA, &root, sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(numCurrFrontier, 0, sizeof(unsigned int));

    // ---- part B: size the grid so that every block is resident at once, and launch (pp. 445-446)
    // TODO 10: how many blocks of BLOCK_DIM threads of bfsCooperativeKernel fit on one SM at
    //          once (an occupancy query), and how many SMs are there (a device property)? The
    //          grid is the product. Then put the address of each kernel argument, in order,
    //          into an array of void*, and launch with cudaLaunchCooperativeKernel, passing all
    //          six arguments. Keep its return value in `err`, and delete `written` and its `if`.
    cudaError_t err = cudaSuccess;
    bool written = false;
    if(!written) {
        return BFS_COOP_NOT_WRITTEN;
    }
    // ---- end of part B
    if(err != cudaSuccess) {
        return BFS_COOP_LAUNCH_ERROR;
    }
    cudaDeviceSynchronize();
    return BFS_COOP_DONE;
}
