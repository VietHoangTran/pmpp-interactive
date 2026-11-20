// Section 18.3's listings: the book's five BFS kernels (Figs. 18.6, 18.8, 18.10, 18.12 and
// 18.14) as printed and as this course corrects them, and the eight visit forms of the
// "Two threads, one neighbour" stepper. The lesson loads this file at run time and shows the
// text between "// region: <name>" and "// endregion".
//
// What is ours, not the page's:
//  - The book never defines its graph structs or UNVISITED. The structs below hold the arrays
//    under the book's names. UNVISITED is UINT_MAX from <climits>: p. 433 prints
//    cuda::std::numeric_limits<unsigned>, which names the trait without its max()
//    (notes/today/ch18-facts.md A9), and the course's emulator has no <cuda/std/limits>.
//  - The book calls all five kernels bfs_kernel. Each figure sits in its own namespace here so
//    that they can share one file; inside a region the book's line numbers hold.
//  - The corrected Figs. 18.6, 18.8 and 18.10 read and write level[] and the flag through
//    relaxed atomics, the "conservative" choice of pp. 433-434: the plain accesses are a data
//    race, which CUDA's memory model leaves undefined even when the writes are idempotent
//    (facts, "Already settled": ch10 A1, ch14 A9). Line for line, only those accesses change.
//  - The corrected Fig. 18.14 adds a HIP branch with the legacy atomicCAS, which is strong on
//    HIP, while libhipcxx lowers compare_exchange_strong to the weak built-in at device scope
//    (B4). Fig. 18.12 needs no change.
//  - The visit forms (visitPlain ... visitWeakCorrect) are ours: Fig. 18.6's check-then-label,
//    Fig. 18.14's compare-and-swap and six variations, five of them wrong on purpose.
//    app/src/sim/bfsvisit.ts models each one; its listing is this file's text.
// Checked by: the CUDA front-end check, and app/src/chapters/ch18/S183.snippets.test.ts, which
// pins every region and runs check/s183_check.cu on the course's emulator.
#include <climits>
#include <cuda/atomic>

#define UNVISITED UINT_MAX

// region: graphs
struct CSRGraph { unsigned int numVertices; unsigned int* srcPtrs; unsigned int* dst; };
struct CSCGraph { unsigned int numVertices; unsigned int* dstPtrs; unsigned int* src; };
struct COOGraph { unsigned int numEdges; unsigned int* src; unsigned int* dst; };
// endregion

// region: relaxed
// A relaxed atomic load and store: no ordering is asked for, and there is no data race.
__device__ inline unsigned int loadRelaxed(unsigned int& x) {
    return cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(x).load(cuda::memory_order_relaxed);
}
__device__ inline void storeRelaxed(unsigned int& x, unsigned int v) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>(x).store(v, cuda::memory_order_relaxed);
}
// endregion

// ---------------------------------------------------------------- Fig. 18.6: push
namespace fig18_6_printed {
// region: fig18_6_printed
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,
                unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x*blockDim.x + threadIdx.x;
    if(vertex < csrGraph.numVertices) {
        if(level[vertex] == currLevel - 1) {
            for(unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if(level[neighbor] == UNVISITED) {
                    level[neighbor] = currLevel;
                    *newVertexVisited = 1;
                }
            }
        }
    }
}
// endregion
}

namespace fig18_6_fixed {
// region: fig18_6_fixed
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,
                unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x*blockDim.x + threadIdx.x;
    if(vertex < csrGraph.numVertices) {
        if(loadRelaxed(level[vertex]) == currLevel - 1) {
            for(unsigned int edge = csrGraph.srcPtrs[vertex];
                    edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = csrGraph.dst[edge];
                if(loadRelaxed(level[neighbor]) == UNVISITED) {
                    storeRelaxed(level[neighbor], currLevel);
                    storeRelaxed(*newVertexVisited, 1);
                }
            }
        }
    }
}
// endregion
}

// ---------------------------------------------------------------- Fig. 18.8: pull
namespace fig18_8_printed {
// region: fig18_8_printed
__global__ void bfs_kernel(CSCGraph cscGraph, unsigned int* level,
                unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x*blockDim.x + threadIdx.x;
    if(vertex < cscGraph.numVertices) {
        if(level[vertex] == UNVISITED) {
            for(unsigned int edge = cscGraph.dstPtrs[vertex];
                    edge < cscGraph.dstPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = cscGraph.src[edge];
                if(level[neighbor] == currLevel - 1) {
                    level[vertex] = currLevel;
                    *newVertexVisited = 1;
                    break;
                }
            }
        }
    }
}
// endregion
}

namespace fig18_8_fixed {
// region: fig18_8_fixed
__global__ void bfs_kernel(CSCGraph cscGraph, unsigned int* level,
                unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int vertex = blockIdx.x*blockDim.x + threadIdx.x;
    if(vertex < cscGraph.numVertices) {
        if(loadRelaxed(level[vertex]) == UNVISITED) {
            for(unsigned int edge = cscGraph.dstPtrs[vertex];
                    edge < cscGraph.dstPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = cscGraph.src[edge];
                if(loadRelaxed(level[neighbor]) == currLevel - 1) {
                    storeRelaxed(level[vertex], currLevel);
                    storeRelaxed(*newVertexVisited, 1);
                    break;
                }
            }
        }
    }
}
// endregion
}

// ---------------------------------------------------------------- Fig. 18.10: edge-centric
namespace fig18_10_printed {
// region: fig18_10_printed
__global__ void bfs_kernel(COOGraph cooGraph, unsigned int* level,
                unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int edge = blockIdx.x*blockDim.x + threadIdx.x;
    if(edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.src[edge];
        if(level[vertex] == currLevel - 1) {
            unsigned int neighbor = cooGraph.dst[edge];
            if(level[neighbor] == UNVISITED) {
                level[neighbor] = currLevel;
                *newVertexVisited = 1;
            }
        }
    }
}
// endregion
}

namespace fig18_10_fixed {
// region: fig18_10_fixed
__global__ void bfs_kernel(COOGraph cooGraph, unsigned int* level,
                unsigned int* newVertexVisited, unsigned int currLevel) {
    unsigned int edge = blockIdx.x*blockDim.x + threadIdx.x;
    if(edge < cooGraph.numEdges) {
        unsigned int vertex = cooGraph.src[edge];
        if(loadRelaxed(level[vertex]) == currLevel - 1) {
            unsigned int neighbor = cooGraph.dst[edge];
            if(loadRelaxed(level[neighbor]) == UNVISITED) {
                storeRelaxed(level[neighbor], currLevel);
                storeRelaxed(*newVertexVisited, 1);
            }
        }
    }
}
// endregion
}

// ---------------------------------------------------------------- Figs. 18.12 and 18.14: frontiers
namespace fig18_12_printed {
// region: fig18_14_printed
__device__ inline unsigned int visitVertexAtomically(unsigned int vertex,
    unsigned int*level, unsigned int currLevel) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
        level_ref(level[vertex]);
    unsigned int unvisited = UNVISITED;
    unsigned int visitSuccess = level_ref.compare_exchange_strong(unvisited,
        currLevel, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
    return visitSuccess;
}
// endregion

// region: fig18_12_printed
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,
            unsigned int* prevFrontier, unsigned int* currFrontier,
            unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
            unsigned int currLevel) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for(unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if(visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    numCurrFrontier_ref(*numCurrFrontier);
                unsigned int currFrontierIdx =
                    numCurrFrontier_ref.fetch_add(1, cuda::memory_order_relaxed);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}
// endregion
}

namespace fig18_12_fixed {
// region: fig18_14_fixed
__device__ inline unsigned int visitVertexAtomically(unsigned int vertex,
    unsigned int*level, unsigned int currLevel) {
#if defined(__HIPCC__)
    // HIP: libhipcxx makes compare_exchange_strong weak at device scope; atomicCAS is strong.
    return atomicCAS(&level[vertex], UNVISITED, currLevel) == UNVISITED;
#else
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
        level_ref(level[vertex]);
    unsigned int unvisited = UNVISITED;
    unsigned int visitSuccess = level_ref.compare_exchange_strong(unvisited,
        currLevel, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
    return visitSuccess;
#endif
}
// endregion

// region: fig18_12_fixed
__global__ void bfs_kernel(CSRGraph csrGraph, unsigned int* level,
            unsigned int* prevFrontier, unsigned int* currFrontier,
            unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
            unsigned int currLevel) {
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for(unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if(visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                    numCurrFrontier_ref(*numCurrFrontier);
                unsigned int currFrontierIdx =
                    numCurrFrontier_ref.fetch_add(1, cuda::memory_order_relaxed);
                currFrontier[currFrontierIdx] = neighbor;
            }
        }
    }
}
// endregion
}

// ---------------------------------------------------------------- the stepper's visit forms
// region: visit_helpers
// Fig. 18.12 lines 12 to 16 as one call: claim a slot with fetch_add, then fill it.
__device__ inline void append(unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int v) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> count(*numCurrFrontier);
    currFrontier[count.fetch_add(1, cuda::memory_order_relaxed)] = v;
}
// Fig. 18.14's compare-and-swap, and the weak form, the only one Metal has. On a failure both
// load the value they found into e.
__device__ inline bool casStrong(unsigned int* p, unsigned int& e, unsigned int desired) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ref(*p);
    return ref.compare_exchange_strong(e, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
}
__device__ inline bool casWeak(unsigned int* p, unsigned int& e, unsigned int desired) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> ref(*p);
    return ref.compare_exchange_weak(e, desired, cuda::memory_order_relaxed, cuda::memory_order_relaxed);
}
// endregion

// region: visit_plain
__device__ void visitPlain(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        if (level[neighbor] == UNVISITED) {
            level[neighbor] = currLevel;
        }
    }
}
// endregion

// region: visit_plain_append
__device__ void visitPlainAppend(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        if (level[neighbor] == UNVISITED) {
            level[neighbor] = currLevel;
            append(currFrontier, numCurrFrontier, neighbor);
        }
    }
}
// endregion

// region: visit_cas
__device__ void visitCAS(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        unsigned int e = UNVISITED;
        if (casStrong(&level[neighbor], e, currLevel)) {
            append(currFrontier, numCurrFrontier, neighbor);
        }
    }
}
// endregion

// region: visit_hoisted
__device__ void visitHoisted(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    unsigned int e = UNVISITED;
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        if (casStrong(&level[neighbor], e, currLevel)) {
            append(currFrontier, numCurrFrontier, neighbor);
        }
    }
}
// endregion

// region: visit_weak
__device__ void visitWeak(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        unsigned int e = UNVISITED;
        if (casWeak(&level[neighbor], e, currLevel)) {
            append(currFrontier, numCurrFrontier, neighbor);
        }
    }
}
// endregion

// region: visit_weak_loop
__device__ void visitWeakLoop(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        unsigned int e = UNVISITED;
        while (!casWeak(&level[neighbor], e, currLevel)) {}
        append(currFrontier, numCurrFrontier, neighbor);
    }
}
// endregion

// region: visit_weak_reset
__device__ void visitWeakReset(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        unsigned int e = UNVISITED;
        while (!casWeak(&level[neighbor], e, currLevel)) e = UNVISITED;
        append(currFrontier, numCurrFrontier, neighbor);
    }
}
// endregion

// region: visit_weak_correct
__device__ void visitWeakCorrect(CSRGraph g, unsigned int vertex, unsigned int* level,
        unsigned int* currFrontier, unsigned int* numCurrFrontier, unsigned int currLevel) {
    for (unsigned int edge = g.srcPtrs[vertex]; edge < g.srcPtrs[vertex + 1]; ++edge) {
        unsigned int neighbor = g.dst[edge];
        unsigned int e = UNVISITED;
        bool won;
        while (!(won = casWeak(&level[neighbor], e, currLevel)) && e == UNVISITED) {}
        if (won) append(currFrontier, numCurrFrontier, neighbor);
    }
}
// endregion
