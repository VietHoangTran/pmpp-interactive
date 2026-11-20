// Lab C.1 starter (CUDA; also built for HIP and for the CPU emulator).
// Three ways to feed one kernel. Book: appendix C, sections C.2 (pp. 624-625),
// C.3 (p. 625) and C.6 (pp. 626-627, Fig. C.1).
//
// The kernel and its launcher are provided and correct. Row r of the result is
// row ids[r] of a table, added up over `passes` launches: an embedding lookup in
// the book's sense (ch. 20, p. 480), repeated so that one feed can read its table
// more than once. The kernel, the passes and the experiment are ours.
//
// Your part is the host side of three feed paths, below the line. Each one does,
// in its own kind of memory, what Fig. C.1 does: get the table written, run the
// kernel, hand the result to CPU code. The contract (feeding.h):
//   - allocate the table, the ids and the result in this path's memory;
//   - call job->produce(table, job->ctx) exactly once to write the table;
//   - give the ids the values of job->ids_h, and start the result at zero;
//   - call launchGatherAdd job->passes times, then make sure the last pass is done;
//   - call job->consume(out, job->ctx) exactly once, with a pointer the CPU may read;
//   - free what you allocated, and return 0. On any CUDA error, print it and
//     return 1: the CHECK macro below does both.
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "../feeding.h"

// The error-checking sidebar of ch. 2 (p. 32) as a macro: report, and make the
// enclosing function return 1. (On an error, allocations made before it leak.)
#define CHECK(call)                                                      \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// ---- provided -------------------------------------------------------------------
// One thread per output element: out[r*cols + c] += table[ids[r]*cols + c].
// Sizes are below 2^31 elements (the driver keeps them there), so int indices do.
__global__ void gatherAddKernel(const float* table, const int* ids, float* out, int count, int cols) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count * cols) {
        int r = i / cols;
        int c = i % cols;
        out[i] += table[ids[r] * cols + c];
    }
}

// What the launcher was last given, for the driver's structural test.
static const void* g_lastTable = nullptr;
static int g_launches = 0;

void launchGatherAdd(const float* table, const int* ids, float* out, int count, int cols) {
    g_lastTable = table;
    ++g_launches;
    const int threadsPerBlock = 256;
    int n = count * cols;
    gatherAddKernel<<<(n + threadsPerBlock - 1) / threadsPerBlock, threadsPerBlock>>>(table, ids, out, count, cols);
}

FeedRecord feedRecord() {
    FeedRecord r;
    r.table = g_lastTable;
    r.launches = g_launches;
    r.tablePrivate = -1;
    r.gpu_copy_s = -1;
    r.gpu_kernel_s = -1;
    return r;
}
void feedResetRecord() {
    g_lastTable = nullptr;
    g_launches = 0;
}

// ---- your code below ------------------------------------------------------------

// 1. The book's simple model (p. 623): device memory, and copies both ways.
int feedExplicit(const FeedJob* job) {
    // TODO 1: let produce write the table into ordinary host memory, the kind
    //         fread fills. The kernel must not read that memory.
    // TODO 2: give the table, the ids and the result a home in device memory.
    //         Fill the first two from the host. Device memory starts as garbage.
    // TODO 3: the passes. A launch does not report its own errors.
    // TODO 4: bring the result to the host and consume it. Which call here waits
    //         for the last pass, and why does it?
    // TODO 5: free everything.
    (void)job;
    printf("feedExplicit: not written yet\n");
    return -1;
}

// 2. Zero-copy, today called mapped memory (C.2): the kernel reads and writes
//    host memory in place. Read p. 624 for the book's procedure, then
//    notes/today/appC-facts.md A3 for what changed since 2009: the structural
//    test expects the kernel to receive the very pointer produce wrote through.
int feedMapped(const FeedJob* job) {
    // TODO 1: allocate all three arrays as pinned host memory the device can
    //         reach (cudaHostAlloc and its flag).
    // TODO 2: produce, ids, a zero result: all written by the CPU, in place.
    // TODO 3: the passes.
    // TODO 4: no copy back follows. What, then, makes it safe to consume?
    // TODO 5: free with the call that matches the allocation.
    (void)job;
    printf("feedMapped: not written yet\n");
    return -1;
}

// 3. Unified Memory (C.6): Fig. C.1, with the table as the file.
int feedManaged(const FeedJob* job) {
    // TODO 1: allocate all three arrays as managed memory. Pass the attach flag
    //         explicitly (cudaMemAttachGlobal). AMD asks for managed sizes in
    //         multiples of 4 KB (appC-facts B5).
    // TODO 2: produce, ids, a zero result, through the same pointers the kernel
    //         will get.
    // TODO 3: the passes.
    // TODO 4: the line Fig. C.1 puts between the kernel and use_data, then consume.
    // TODO 5: free.
    (void)job;
    printf("feedManaged: not written yet\n");
    return -1;
}
