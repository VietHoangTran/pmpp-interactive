// Lab 18.1 reference solution (CUDA; the same file is built for HIP and for the two CPU
// emulators). Breadth-first search with frontiers, book sections 18.5 to 18.7 (pp. 438-448).
//   visitVertexAtomically   Fig. 18.14 (p. 441): label a vertex exactly once, by compare-and-swap.
//   bfsPrivatisedKernel     Fig. 18.15 (p. 443), part A: one launch per level. Each block
//                           collects the vertices it finds in a private frontier in shared
//                           memory, reserves its range of the public frontier with one atomic,
//                           and copies them there with coalesced stores.
//   bfsCooperativeKernel    Fig. 18.17 (p. 447), part B: every level in one launch, with
//                           grid-wide barriers between levels.
//
// The graph is in CSR form (section 18.1, p. 427): the out-neighbours of vertex v are
// dst[srcPtrs[v]] .. dst[srcPtrs[v + 1] - 1]. level[v] is v's BFS level, UNVISITED until found.
//
// Where this file differs from the page:
//   - Raw pointers and sizes (srcPtrs, dst, numVertices) instead of the book's CSRGraph, which
//     the chapter never defines. The kernel names are ours: every listing is called bfs_kernel.
//   - UNVISITED is UINT_MAX from <climits>. The page's `cuda::std::numeric_limits<unsigned>`
//     (p. 433) names the trait without ::max() (notes/today/ch18-facts.md A9), and the CPU
//     emulator has no <cuda/std/limits>. The value is the same.
//   - Fig. 18.15 line 29 (and Fig. 18.17 line 35) put the private counter back to the capacity
//     with a plain store while other threads of the block update it with fetch_add. That is a
//     data race in CUDA's memory model (A7). Here it is an atomic store through the same
//     reference; the arithmetic is unchanged (E1).
//   - Fig. 18.17 declares its __shared__ variables inside the level loop. They are hoisted to
//     the top of the kernel here: a __shared__ variable lives as long as the block, not one
//     pass of the loop, so the declaration says where the storage really is. The reset at the
//     top of each level is what gives each level an empty private frontier.
//   - grid.thread_rank() and grid.num_threads() return unsigned long long on CUDA (A4) and
//     32-bit values on HIP (B2). They are cast to unsigned int once; the page narrows them
//     silently in the loop header.
//   - cudaLaunchCooperativeKernel gets all six arguments. The page passes four, which CUDA
//     accepts (its last two parameters have defaults) and HIP does not (A1, B1).
//   - The page shows no host loop for Fig. 18.15. launchBfsPrivatised is ours.
//   - On HIP the atomics are the legacy atomicCAS, atomicAdd and atomicExch: see below.
#include <climits>

// cuda::atomic_ref comes from <cuda/atomic>. On HIP that header is libhipcxx, which needs ROCm
// 10.1 or later and its own CMake target (notes/today/infra-facts.md B2), and whose
// compare_exchange_strong is lowered to the weak built-in at block and device scope
// (notes/today/ch18-facts.md B4). So under __HIPCC__ the legacy functions stand in, as in lab
// 12.1: atomicCAS (strong; returns the old value), atomicAdd and atomicExch, all relaxed and
// valid on shared and global memory (infra-facts B1).
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
// the block so that the commit loop must take more than one step per thread. BLOCK_DIM is a
// multiple of 64, so that every warp is whole on every vendor (AMD wavefronts can be 64 lanes,
// notes/today/ch05-facts.md B5).
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

// region: visit
// Fig. 18.14. True for exactly one of the threads that try to visit the same vertex.
__device__ inline unsigned int visitVertexAtomically(unsigned int vertex, unsigned int* level,
                                                     unsigned int currLevel) {
#if defined(__HIPCC__)
    return atomicCAS(&level[vertex], UNVISITED, currLevel) == UNVISITED;
#else
    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> level_ref(level[vertex]);
    unsigned int unvisited = UNVISITED;   // fresh on every call: a failed CAS writes the winner's level here
    unsigned int visitSuccess = level_ref.compare_exchange_strong(unvisited, currLevel,
                                                                  cuda::memory_order_relaxed,
                                                                  cuda::memory_order_relaxed);
    return visitSuccess;
#endif
}
// endregion

// region: privatised
__global__ void bfsPrivatisedKernel(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                    unsigned int* level, unsigned int* prevFrontier, unsigned int* currFrontier,
                                    unsigned int numPrevFrontier, unsigned int* numCurrFrontier,
                                    unsigned int currLevel) {
    // Initialize privatized frontier
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    if(threadIdx.x == 0) {
        numCurrFrontier_s = 0;
    }
    __syncthreads();

    // Perform BFS
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < numPrevFrontier) {
        unsigned int vertex = prevFrontier[i];
        for(unsigned int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; ++edge) {
            unsigned int neighbor = dst[edge];
            if(visitVertexAtomically(neighbor, level, currLevel)) {
#if defined(__HIPCC__)
                unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1u);
#else
                cuda::atomic_ref<unsigned int, cuda::thread_scope_block> numCurrFrontier_s_ref(numCurrFrontier_s);
                unsigned int currFrontierIdx_s = numCurrFrontier_s_ref.fetch_add(1, cuda::memory_order_relaxed);
#endif
                if(currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                    currFrontier_s[currFrontierIdx_s] = neighbor;
                } else {
                    // Overflow: put the counter back to the capacity, with an atomic store (the
                    // page's plain store is a data race), and append to the public frontier.
#if defined(__HIPCC__)
                    atomicExch(&numCurrFrontier_s, (unsigned int)PRIVATE_FRONTIER_CAPACITY);
                    unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1u);
#else
                    numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY, cuda::memory_order_relaxed);
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_device> numCurrFrontier_ref(*numCurrFrontier);
                    unsigned int currFrontierIdx = numCurrFrontier_ref.fetch_add(1, cuda::memory_order_relaxed);
#endif
                    currFrontier[currFrontierIdx] = neighbor;
                }
            }
        }
    }
    __syncthreads();

    // Allocate in global frontier: one device-scope atomic per block
    __shared__ unsigned int currFrontierStartIdx;
    if(threadIdx.x == 0) {
#if defined(__HIPCC__)
        currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
#else
        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> numCurrFrontier_ref(*numCurrFrontier);
        currFrontierStartIdx = numCurrFrontier_ref.fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
#endif
    }
    __syncthreads();

    // Commit to global frontier: consecutive threads write consecutive slots
    for(unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s;
        currFrontierIdx_s += blockDim.x) {
        unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
        currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
    }
}
// endregion

// region: privatised-host
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
// endregion

// region: cooperative
// Part B: every level in one launch. Launch only through launchBfsCooperative: grid.sync()
// needs a cooperative launch, and every block of the grid must be resident at once (p. 445).
__global__ void bfsCooperativeKernel(const unsigned int* srcPtrs, const unsigned int* dst, unsigned int numVertices,
                                     unsigned int* level, unsigned int* prevFrontier, unsigned int* currFrontier,
                                     unsigned int numPrevFrontier, unsigned int* numCurrFrontier) {
    __shared__ unsigned int currFrontier_s[PRIVATE_FRONTIER_CAPACITY];
    __shared__ unsigned int numCurrFrontier_s;
    __shared__ unsigned int currFrontierStartIdx;

    cg::grid_group grid = cg::this_grid();
    const unsigned int gridRank = (unsigned int)grid.thread_rank();
    const unsigned int gridThreads = (unsigned int)grid.num_threads();

    for(unsigned int currLevel = 1; numPrevFrontier > 0; ++currLevel) {
        // Initialize privatized frontier
        if(threadIdx.x == 0) {
            numCurrFrontier_s = 0;
        }
        __syncthreads();

        // Perform BFS: the grid is smaller than the frontier, so each thread strides through it
        for(unsigned int i = gridRank; i < numPrevFrontier; i += gridThreads) {
            unsigned int vertex = prevFrontier[i];
            for(unsigned int edge = srcPtrs[vertex]; edge < srcPtrs[vertex + 1]; ++edge) {
                unsigned int neighbor = dst[edge];
                if(visitVertexAtomically(neighbor, level, currLevel)) {
#if defined(__HIPCC__)
                    unsigned int currFrontierIdx_s = atomicAdd(&numCurrFrontier_s, 1u);
#else
                    cuda::atomic_ref<unsigned int, cuda::thread_scope_block> numCurrFrontier_s_ref(numCurrFrontier_s);
                    unsigned int currFrontierIdx_s = numCurrFrontier_s_ref.fetch_add(1, cuda::memory_order_relaxed);
#endif
                    if(currFrontierIdx_s < PRIVATE_FRONTIER_CAPACITY) {
                        currFrontier_s[currFrontierIdx_s] = neighbor;
                    } else {
#if defined(__HIPCC__)
                        atomicExch(&numCurrFrontier_s, (unsigned int)PRIVATE_FRONTIER_CAPACITY);
                        unsigned int currFrontierIdx = atomicAdd(numCurrFrontier, 1u);
#else
                        numCurrFrontier_s_ref.store(PRIVATE_FRONTIER_CAPACITY, cuda::memory_order_relaxed);
                        cuda::atomic_ref<unsigned int, cuda::thread_scope_device> numCurrFrontier_ref(*numCurrFrontier);
                        unsigned int currFrontierIdx = numCurrFrontier_ref.fetch_add(1, cuda::memory_order_relaxed);
#endif
                        currFrontier[currFrontierIdx] = neighbor;
                    }
                }
            }
        }
        __syncthreads();

        // Allocate in global frontier
        if(threadIdx.x == 0) {
#if defined(__HIPCC__)
            currFrontierStartIdx = atomicAdd(numCurrFrontier, numCurrFrontier_s);
#else
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> numCurrFrontier_ref(*numCurrFrontier);
            currFrontierStartIdx = numCurrFrontier_ref.fetch_add(numCurrFrontier_s, cuda::memory_order_relaxed);
#endif
        }
        __syncthreads();

        // Commit to global frontier
        for(unsigned int currFrontierIdx_s = threadIdx.x; currFrontierIdx_s < numCurrFrontier_s;
            currFrontierIdx_s += blockDim.x) {
            unsigned int currFrontierIdx = currFrontierStartIdx + currFrontierIdx_s;
            currFrontier[currFrontierIdx] = currFrontier_s[currFrontierIdx_s];
        }

        // Swap frontiers: three grid-wide barriers, each with its own job (p. 448).
        grid.sync();                                // 1. every block has appended and committed
        numPrevFrontier = *numCurrFrontier;         //    so every thread reads the final count
        grid.sync();                                // 2. every thread has read it
        if(gridRank == 0) {
            *numCurrFrontier = 0;                   //    so it can be reset
        }
        grid.sync();                                // 3. the reset is visible before anyone appends
        unsigned int* tmp = prevFrontier;
        prevFrontier = currFrontier;
        currFrontier = tmp;
    }
}
// endregion

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

    // region: sizing
    // ---- part B: size the grid so that every block is resident at once, and launch (pp. 445-446)
    int numBlocksPerSM = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocksPerSM, bfsCooperativeKernel, BLOCK_DIM, 0);
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    int numSMs = deviceProp.multiProcessorCount;
    unsigned int numBlocks = numSMs*numBlocksPerSM;

    unsigned int numPrevFrontier = 1;
    void* kernelArgs[] = { (void*)&srcPtrs, (void*)&dst, (void*)&numVertices, (void*)&level,
                           (void*)&frontierA, (void*)&frontierB, (void*)&numPrevFrontier,
                           (void*)&numCurrFrontier };
    cudaError_t err = cudaLaunchCooperativeKernel((void*)bfsCooperativeKernel, numBlocks, BLOCK_DIM,
                                                  kernelArgs, 0, 0);
    // ---- end of part B
    // endregion
    if(err != cudaSuccess) {
        return BFS_COOP_LAUNCH_ERROR;
    }
    cudaDeviceSynchronize();
    return BFS_COOP_DONE;
}
