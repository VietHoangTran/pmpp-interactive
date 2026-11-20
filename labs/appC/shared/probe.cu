// Lab C.1, provided (CUDA; also built for HIP and for the CPU emulator).
// Two instruments for the benchmark. Neither is in the book.
//
//  - queryManagedParadigm: which of NVIDIA's four unified-memory paradigms this
//    device has, from three device attributes (notes/today/appC-facts.md A14).
//    AMD's HIP has the same attributes with CUDA's meanings (B5). The names of the
//    attributes differ, and pmpp/vendors/hip.h does not map them, so the HIP build
//    calls HIP by its own names here.
//  - probeBandwidth: how fast pinned host memory reaches the device over the
//    interconnect, and how fast the device copies within its own memory. The
//    book's rule of thumb is that the first is under a tenth of the second on a
//    PCIe system (p. 625; A4 says where that still holds). Measure it.
//
// The emulator and the CUDA front-end check have none of the three attributes,
// so on those the paradigm query answers "emulator" without asking.
#include <cstdio>
#include <cstring>

#include "../lab1_feeding/feeding.h"

#define PROBE_CHECK(call)                                                \
    do {                                                                 \
        cudaError_t err = (call);                                        \
        if (err != cudaSuccess) {                                        \
            printf("%s in %s at line %d\n", cudaGetErrorString(err),     \
                   __FILE__, __LINE__);                                  \
            return 1;                                                    \
        }                                                                \
    } while (0)

// A14's table: ConcurrentManagedAccess 0 is the limited paradigm (Windows, WSL,
// some Tegra); 1 with PageableMemoryAccess 0 is full support for managed
// allocations only; both 1 is full support for all memory, through software
// coherence (HMM) when PageableMemoryAccessUsesHostPageTables is 0 and hardware
// coherence (ATS) when it is 1.
static const char* paradigmName(int cma, int pma, int uhpt) {
    if (cma == 0) return "limited";
    if (pma == 0) return "full-managed";
    return uhpt ? "full-ats" : "full-hmm";
}

const char* queryManagedParadigm(int* concurrentManaged, int* pageableAccess, int* usesHostPageTables) {
    *concurrentManaged = *pageableAccess = *usesHostPageTables = -1;
#if defined(PMPP_EMU) || defined(PMPP_CUDA_FRONTEND_CHECK)
    return "emulator";
#elif defined(PMPP_BACKEND_HIP)
    // Unverified: written from the ROCm 10.2 header names (B5), not compiled here.
    if (hipDeviceGetAttribute(concurrentManaged, hipDeviceAttributeConcurrentManagedAccess, 0) != hipSuccess ||
        hipDeviceGetAttribute(pageableAccess, hipDeviceAttributePageableMemoryAccess, 0) != hipSuccess ||
        hipDeviceGetAttribute(usesHostPageTables, hipDeviceAttributePageableMemoryAccessUsesHostPageTables, 0) != hipSuccess)
        return "unknown";
    return paradigmName(*concurrentManaged, *pageableAccess, *usesHostPageTables);
#else
    if (cudaDeviceGetAttribute(concurrentManaged, cudaDevAttrConcurrentManagedAccess, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(pageableAccess, cudaDevAttrPageableMemoryAccess, 0) != cudaSuccess ||
        cudaDeviceGetAttribute(usesHostPageTables, cudaDevAttrPageableMemoryAccessUsesHostPageTables, 0) != cudaSuccess)
        return "unknown";
    return paradigmName(*concurrentManaged, *pageableAccess, *usesHostPageTables);
#endif
}

int probeBandwidth(size_t bytes, int reps, double* link_gb_per_s, double* device_gb_per_s) {
    void* host = nullptr;
    void* a = nullptr;
    void* b = nullptr;
    // Pinned host memory, which NVIDIA says copies "with much higher bandwidth than
    // pageable memory obtained with functions such as ::malloc()" (ch02-facts A16).
    // feedExplicit's staging buffer is malloc memory, so its copy may be slower.
    PROBE_CHECK(cudaMallocHost(&host, bytes));
    PROBE_CHECK(cudaMalloc(&a, bytes));
    PROBE_CHECK(cudaMalloc(&b, bytes));
    memset(host, 1, bytes);

    cudaEvent_t start, stop;
    PROBE_CHECK(cudaEventCreate(&start));
    PROBE_CHECK(cudaEventCreate(&stop));
    float bestLink = 1e30f, bestDevice = 1e30f;
    for (int r = 0; r <= reps; ++r) {   // r == 0 is an untimed warm-up
        float ms = 0;
        PROBE_CHECK(cudaEventRecord(start));
        PROBE_CHECK(cudaMemcpy(a, host, bytes, cudaMemcpyHostToDevice));
        PROBE_CHECK(cudaEventRecord(stop));
        PROBE_CHECK(cudaEventSynchronize(stop));
        PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (r > 0 && ms < bestLink) bestLink = ms;

        PROBE_CHECK(cudaEventRecord(start));
        PROBE_CHECK(cudaMemcpy(b, a, bytes, cudaMemcpyDeviceToDevice));
        PROBE_CHECK(cudaEventRecord(stop));
        PROBE_CHECK(cudaEventSynchronize(stop));
        PROBE_CHECK(cudaEventElapsedTime(&ms, start, stop));
        if (r > 0 && ms < bestDevice) bestDevice = ms;
    }
    *link_gb_per_s = (double)bytes / (bestLink * 1e-3) / 1e9;
    *device_gb_per_s = 2.0 * (double)bytes / (bestDevice * 1e-3) / 1e9;   // read plus write

    PROBE_CHECK(cudaEventDestroy(start));
    PROBE_CHECK(cudaEventDestroy(stop));
    PROBE_CHECK(cudaFree(a));
    PROBE_CHECK(cudaFree(b));
    PROBE_CHECK(cudaFreeHost(host));
    return 0;
}
