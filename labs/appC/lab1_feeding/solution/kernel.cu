// Lab C.1 reference solution (CUDA; also built for HIP and for the CPU emulator).
// Three ways to feed one kernel. Book: appendix C, sections C.2 (pp. 624-625),
// C.3 (p. 625) and C.6 (pp. 626-627, Fig. C.1).
//
// The kernel is an embedding lookup in the book's sense (ch. 20, p. 480): row r of
// the result is row ids[r] of a table. It runs `passes` times and adds up, so that
// one feed can read its table more than once. The kernel, the passes and the
// whole experiment are ours; the book shows no code for C.2 and gives no timings.
//
// Where this file differs from the book:
//  - feedMapped hands the cudaHostAlloc pointer straight to the kernel. The book
//    says that pointer cannot be passed and a second one must come from
//    cudaHostGetDevicePointer() (p. 624). That was CUDA 2.2. Under unified
//    addressing NVIDIA documents both the mapped flag and the second pointer as
//    unnecessary for cudaHostAlloc memory (notes/today/appC-facts.md A3); the
//    pointer is the same value on both sides.
//  - feedManaged is Fig. C.1 (p. 627) with the elided launch configuration filled
//    in (it lives in launchGatherAdd), the semicolon after cudaDeviceSynchronize()
//    that the page drops, and every call checked.
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

// 1. The book's simple model (p. 623): the CPU fills host memory, the data is
//    copied to device memory, the kernel runs there, the result is copied back.
int feedExplicit(const FeedJob* job) {
    const size_t tableBytes = feedTableBytes(job), idsBytes = feedIdsBytes(job), outBytes = feedOutBytes(job);

    // Ordinary host memory, where fread would put a file.
    float* table_h = (float*)malloc(tableBytes);
    float* out_h = (float*)malloc(outBytes);
    if (table_h == nullptr || out_h == nullptr) {
        printf("malloc failed in %s at line %d\n", __FILE__, __LINE__);
        free(table_h);
        free(out_h);
        return 1;
    }
    job->produce(table_h, job->ctx);

    float *table_d, *out_d;
    int* ids_d;
    CHECK(cudaMalloc((void**)&table_d, tableBytes));
    CHECK(cudaMalloc((void**)&ids_d, idsBytes));
    CHECK(cudaMalloc((void**)&out_d, outBytes));
    CHECK(cudaMemcpy(table_d, table_h, tableBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(ids_d, job->ids_h, idsBytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemset(out_d, 0, outBytes));   // cudaMalloc memory is not cleared

    for (int p = 0; p < job->passes; ++p) {
        launchGatherAdd(table_d, ids_d, out_d, job->count, job->cols);
        CHECK(cudaGetLastError());
    }
    // Same stream as the launches, and a blocking copy: it starts after the last
    // pass and returns when the result is in out_h.
    CHECK(cudaMemcpy(out_h, out_d, outBytes, cudaMemcpyDeviceToHost));
    job->consume(out_h, job->ctx);

    CHECK(cudaFree(table_d));
    CHECK(cudaFree(ids_d));
    CHECK(cudaFree(out_d));
    free(table_h);
    free(out_h);
    return 0;
}

// 2. Zero-copy, now called mapped memory (C.2; A3): pinned host memory that the
//    kernel reads and writes in place, over the interconnect, on every access.
int feedMapped(const FeedJob* job) {
    const size_t tableBytes = feedTableBytes(job), idsBytes = feedIdsBytes(job), outBytes = feedOutBytes(job);

    float *table, *out;
    int* ids;
    CHECK(cudaHostAlloc((void**)&table, tableBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&ids, idsBytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void**)&out, outBytes, cudaHostAllocMapped));

    // The CPU writes where the kernel will read: no copy anywhere.
    job->produce(table, job->ctx);
    memcpy(ids, job->ids_h, idsBytes);
    memset(out, 0, outBytes);

    // The same pointer values go to the kernel (see the header comment).
    for (int p = 0; p < job->passes; ++p) {
        launchGatherAdd(table, ids, out, job->count, job->cols);
        CHECK(cudaGetLastError());
    }
    // Nothing else waits for the kernel here: no copy back follows to do it.
    CHECK(cudaDeviceSynchronize());
    job->consume(out, job->ctx);

    CHECK(cudaFreeHost(table));
    CHECK(cudaFreeHost(ids));
    CHECK(cudaFreeHost(out));
    return 0;
}

// AMD asks for managed allocations in multiples of 4 KB ("it should be granularity
// of 4KB", appC-facts B5). CUDA states no such rule, and rounding up costs nothing.
static size_t managedBytes(size_t bytes) {
    const size_t granule = 4096;
    return (bytes + granule - 1) / granule * granule;
}

// 3. Unified Memory (C.6), as in Fig. C.1: one pointer for the CPU and the GPU,
//    and the runtime moves pages as each side touches them.
int feedManaged(const FeedJob* job) {
    const size_t tableBytes = feedTableBytes(job), idsBytes = feedIdsBytes(job), outBytes = feedOutBytes(job);

    float *table, *out;
    int* ids;
    CHECK(cudaMallocManaged((void**)&table, managedBytes(tableBytes), cudaMemAttachGlobal));
    CHECK(cudaMallocManaged((void**)&ids, managedBytes(idsBytes), cudaMemAttachGlobal));
    CHECK(cudaMallocManaged((void**)&out, managedBytes(outBytes), cudaMemAttachGlobal));

    job->produce(table, job->ctx);   // Fig. C.1's fread, into managed memory
    memcpy(ids, job->ids_h, idsBytes);
    memset(out, 0, outBytes);        // managed memory is not cleared either (A9)

    for (int p = 0; p < job->passes; ++p) {
        launchGatherAdd(table, ids, out, job->count, job->cols);
        CHECK(cudaGetLastError());
    }
    // Fig. C.1's cudaDeviceSynchronize(). Without it the CPU reads while the GPU
    // may still be writing: a race where ConcurrentManagedAccess is 1, and a
    // segmentation fault where it is 0, which includes every Windows system (A10).
    CHECK(cudaDeviceSynchronize());
    job->consume(out, job->ctx);     // Fig. C.1's use_data

    CHECK(cudaFree(table));
    CHECK(cudaFree(ids));
    CHECK(cudaFree(out));
    return 0;
}
