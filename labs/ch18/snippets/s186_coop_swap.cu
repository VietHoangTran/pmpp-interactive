// Section 18.6's "Three grid barriers": every BFS level in one cooperative launch, with the
// frontier swap of Fig. 18.17 (lines 65-74, p. 447) and Exercise 3's one-barrier form
// (p. 450), plus copies that drop one grid barrier each.
//
// bfsCoop is the swap with its three grid.sync() calls. bfsCoopNoSync1, 2 and 3 each drop
// one. bfsCoopEx3 keeps one counter per level, all zero before the launch, and needs one
// barrier per level; bfsCoopEx3NoSync drops it. residentBlocks() and launchBfsCoop() are
// the book's host code (pp. 445-446).
// app/src/sim/gridbfs.ts models every kernel; app/src/sim/gridbfs.test.ts pins each kernel
// body to the model's text and runs this file on the course's emulator, in its three thread
// orders, through check/s186_coop_check.cu.
//
// Differences from the page:
//   - The private queue of Fig. 18.17 (lines 12-62) is left out: a new vertex is appended
//     straight to the public counter, as in Fig. 18.12. The section's other stepper and the
//     labs page's reference solution have the queue. Without its block barriers a thread runs
//     from one grid barrier to the next without stopping, so that each removed grid barrier
//     can fail on the emulator, which hands the baton from thread to thread.
//   - The header is chosen by vendor: HIP's is <hip/hip_cooperative_groups.h>
//     (notes/today/ch18-facts.md B1; llama.cpp does the same, D2). The names are reached
//     through cg:: rather than the book's `using namespace cooperative_groups;`.
//   - grid.thread_rank() and grid.num_threads() return unsigned long long on CUDA (A4) and
//     32-bit values on HIP (B2). The book's `unsigned int i` narrows them; harmless for any
//     grid that is small enough to be resident.
//   - Every atomic_ref lives only inside the `if` that appends, so none is alive when the
//     counter is read and reset with plain accesses (A5's caveat).
//   - The launch passes all six arguments: HIP's hipLaunchCooperativeKernel has no default
//     arguments, so the book's four-argument call does not compile there (B1).
// region: header
#if defined(__HIPCC__)
#include <hip/hip_cooperative_groups.h>
#else
#include <cooperative_groups.h>
#endif
#include <cuda/atomic>

namespace cg = cooperative_groups;
// endregion

struct CSRGraph {
    unsigned int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
};

constexpr unsigned int UNVISITED = 0xFFFFFFFFu;

// Fig. 18.14 (p. 441).
__device__ inline unsigned int visitVertexAtomically(unsigned int vertex, unsigned int* level,
        unsigned int currLevel) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> level_ref(level[vertex]);
    unsigned int unvisited = UNVISITED;
    unsigned int visitSuccess = level_ref.compare_exchange_strong(unvisited, currLevel,
        cuda::memory_order_relaxed, cuda::memory_order_relaxed);
    return visitSuccess;
}

__global__ void bfsCoop(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier) {
    cg::grid_group grid = cg::this_grid();
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;
                i += grid.num_threads()) {
            unsigned int vertex = prevFrontier[i];
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (visitVertexAtomically(neighbor, level, currLevel)) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
        grid.sync();  // 1: every append is done
        numPrevFrontier = *numCurrFrontier;
        grid.sync();  // 2: everyone has read the size
        if (grid.thread_rank() == 0) {
            *numCurrFrontier = 0;
        }
        grid.sync();  // 3: the reset is visible
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}

__global__ void bfsCoopNoSync1(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier) {
    cg::grid_group grid = cg::this_grid();
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;
                i += grid.num_threads()) {
            unsigned int vertex = prevFrontier[i];
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (visitVertexAtomically(neighbor, level, currLevel)) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
        numPrevFrontier = *numCurrFrontier;
        grid.sync();  // 2: everyone has read the size
        if (grid.thread_rank() == 0) {
            *numCurrFrontier = 0;
        }
        grid.sync();  // 3: the reset is visible
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}

__global__ void bfsCoopNoSync2(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier) {
    cg::grid_group grid = cg::this_grid();
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;
                i += grid.num_threads()) {
            unsigned int vertex = prevFrontier[i];
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (visitVertexAtomically(neighbor, level, currLevel)) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
        grid.sync();  // 1: every append is done
        numPrevFrontier = *numCurrFrontier;
        if (grid.thread_rank() == 0) {
            *numCurrFrontier = 0;
        }
        grid.sync();  // 3: the reset is visible
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}

__global__ void bfsCoopNoSync3(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier) {
    cg::grid_group grid = cg::this_grid();
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;
                i += grid.num_threads()) {
            unsigned int vertex = prevFrontier[i];
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (visitVertexAtomically(neighbor, level, currLevel)) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
        grid.sync();  // 1: every append is done
        numPrevFrontier = *numCurrFrontier;
        grid.sync();  // 2: everyone has read the size
        if (grid.thread_rank() == 0) {
            *numCurrFrontier = 0;
        }
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}

// Exercise 3: numFrontier[L] counts level L's frontier. All are zero before the launch, so
// nothing is ever reset and one barrier per level is enough. The price is a bound on the
// number of levels: numFrontier needs one counter per level.
__global__ void bfsCoopEx3(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numFrontier) {
    cg::grid_group grid = cg::this_grid();
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;
                i += grid.num_threads()) {
            unsigned int vertex = prevFrontier[i];
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (visitVertexAtomically(neighbor, level, currLevel)) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numFrontier_ref(numFrontier[currLevel]);
                    unsigned int currFrontierIdx = numFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
        grid.sync();  // the only barrier: every append is done
        numPrevFrontier = numFrontier[currLevel];
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}

__global__ void bfsCoopEx3NoSync(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numFrontier) {
    cg::grid_group grid = cg::this_grid();
    for (unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        for (unsigned int i = grid.thread_rank(); i < numPrevFrontier;
                i += grid.num_threads()) {
            unsigned int vertex = prevFrontier[i];
            for (unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if (visitVertexAtomically(neighbor, level, currLevel)) {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numFrontier_ref(numFrontier[currLevel]);
                    unsigned int currFrontierIdx = numFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
        numPrevFrontier = numFrontier[currLevel];
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}

// region: launch
// pp. 445-446: launch no more blocks than can be resident at once.
unsigned int residentBlocks(unsigned int numThreadsPerBlock) {
    int numBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, bfsCoop,
        numThreadsPerBlock, 0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int numSMs = deviceProp.multiProcessorCount;
    return (unsigned int)(numSMs * numBlocksPerSM);
}

// p. 446, with all six arguments, so that the same line compiles for HIP.
cudaError_t launchBfsCoop(CSRGraph csrGraph_d, unsigned int* level_d,
        unsigned int* prevFrontier_d, unsigned int* currFrontier_d,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier_d,
        unsigned int numBlocks, unsigned int numThreadsPerBlock) {
    void* kernelArgs[] = { (void*)&csrGraph_d, (void*)&level_d,
        (void*)&prevFrontier_d, (void*)&currFrontier_d,
        (void*)&numPrevFrontier, (void*)&numCurrFrontier_d };
    return cudaLaunchCooperativeKernel((void*)bfsCoop, numBlocks,
        numThreadsPerBlock, kernelArgs, 0, 0);
}
// endregion
