// Section 18.6's stepper: one level of the privatised-frontier BFS of Fig. 18.15 (p. 443),
// corrected, and seven copies that differ from it in one place each.
//
// bfsPrivatised is the corrected kernel. bfsPrivatisedClamp is the other correction.
// bfsPlainStore is the book's line 29 as printed. bfsNoBarrier1, 2 and 3 each drop one
// block barrier, bfsIfCommit commits with an `if` instead of the strided loop, and
// bfsNoOverflowCheck stores every new vertex in the private queue.
// app/src/sim/bfspriv.ts models every kernel; app/src/sim/bfspriv.test.ts pins each kernel
// body to the model's text, runs this file on the course's emulator in its three thread
// orders and on its race flavour through check/s186_privatised_check.cu, and compares
// both with the model.
//
// Differences from the page:
//   - Line 29, `numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;`, is a plain store to a
//     counter other threads of the block are updating with fetch_add at the same time: a
//     data race (notes/today/ch18-facts.md A7). bfsPrivatised stores through the same
//     atomic_ref; bfsPrivatisedClamp drops the store and clamps the count after barrier 2.
//     Both leave the counter at min(N, capacity), which is what the reservation needs (E1).
//   - PRIVATE_FRONTIER_CAPACITY is a template parameter so that one build can try every
//     capacity from 1 to 8. The book makes it a constant and does not give its value.
//   - UNVISITED is spelled 0xFFFFFFFFu, the largest unsigned int. The page's
//     `cuda::std::numeric_limits<unsigned>` is incomplete; the full expression is
//     cuda::std::numeric_limits<unsigned int>::max() from <cuda/std/limits> (A9).
//   - Long statements are broken over two lines so that the stepper's listing fits its
//     column. The names are the book's.
// The book shows no capacity check on the public frontier either: with the
// compare-and-swap each vertex is appended at most once, so numVertices slots suffice.
#include <cuda/atomic>

struct CSRGraph {
    unsigned int numVertices;
    unsigned int* srcPtrs;
    unsigned int* dst;
};

constexpr unsigned int UNVISITED = 0xFFFFFFFFu;

// Fig. 18.14 (p. 441): label the vertex with currLevel if, and only if, nobody has yet.
__device__ inline unsigned int visitVertexAtomically(unsigned int vertex, unsigned int* level,
        unsigned int currLevel) {
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> level_ref(level[vertex]);
    unsigned int unvisited = UNVISITED;
    unsigned int visitSuccess = level_ref.compare_exchange_strong(unvisited, currLevel,
        cuda::memory_order_relaxed, cuda::memory_order_relaxed);
    return visitSuccess;
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsPrivatised(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,
                        cuda::memory_order_relaxed);
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsPrivatisedClamp(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        if (numCurrFrontier_s > PRIVATE_FRONTIER_CAPACITY)
            numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsPlainStore(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s = PRIVATE_FRONTIER_CAPACITY;
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsNoBarrier1(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,
                        cuda::memory_order_relaxed);
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsNoBarrier2(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,
                        cuda::memory_order_relaxed);
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsNoBarrier3(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,
                        cuda::memory_order_relaxed);
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsIfCommit(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                if (currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY,
                        cuda::memory_order_relaxed);
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
                        numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref
                        .fetch_add(1, cuda::memory_order_relaxed);
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    if (threadIdx.x < numCurrFrontier_s)
        currFrontier[currFrontierStartIdx + threadIdx.x] =
            currFrontier_s[threadIdx.x];
}

template <unsigned int PRIVATE_FRONTIER_CAPACITY>
__global__ void bfsNoOverflowCheck(CSRGraph csrGraph, unsigned int* level,
        unsigned int* prevFrontier, unsigned int* currFrontier,
        unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
        unsigned int currLevel) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if (threadIdx.x == 0) numCurrFrontier_s = 0;
    __syncthreads();  // barrier 1
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for (unsigned int edge = csrGraph.srcPtrs[vertex];
                edge < csrGraph.srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = csrGraph.dst[edge];
            if (visitVertexAtomically(neighbor, level, currLevel)) {
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block>
                    numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref
                    .fetch_add(1, cuda::memory_order_relaxed);
                currFrontier_s[currFrontierIdx_s] = neighbor;
            }
        }
    }
    __syncthreads();  // barrier 2
    __shared__ unsigned int currFrontierStartIdx;
    if (threadIdx.x == 0) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device>
            numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref
            .fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
    }
    __syncthreads();  // barrier 3
    for (unsigned int currFrontierIdx_s = threadIdx.x;
            currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx =
            currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}
