// Self-test, positive control for the race detector: the histogram of
// race_hist_plain.cu with atomic updates. Correct everywhere, and it must NOT
// be reported by ThreadSanitizer.
#include <cuda/atomic>
#include "selftest_cu.h"

__global__ void kHistAtomic(const unsigned char* data, unsigned* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(&bins[data[i] / 64], 1u);
}

__global__ void kHistAtomicRef(const unsigned char* data, unsigned* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        cuda::atomic_ref<unsigned, cuda::thread_scope_device> bin(bins[data[i] / 64]);
        bin.fetch_add(1u, cuda::memory_order_relaxed);
    }
}

// Per-block private histogram in shared memory, merged with one atomic per bin.
__global__ void kHistPrivate(const unsigned char* data, unsigned* bins, int n) {
    __shared__ unsigned local[4];
    int t = threadIdx.x, i = blockIdx.x * blockDim.x + t;
    if (t < 4) local[t] = 0;
    __syncthreads();
    if (i < n) atomicAdd(&local[data[i] / 64], 1u);
    __syncthreads();
    if (t < 4) atomicAdd(&bins[t], local[t]);
}

static std::string run(int which) {
    const int n = 4096, T = 64;
    std::vector<unsigned char> h(n);
    unsigned want[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) { h[i] = (unsigned char)((i * 131 + 7) % 256); want[h[i] / 64]++; }
    StBuf<unsigned char> data(h); StBuf<unsigned> bins(4, 0u);
    if (which == 0) kHistAtomic<<<n / T, T>>>(data.dev, bins.dev, n);
    else if (which == 1) kHistAtomicRef<<<n / T, T>>>(data.dev, bins.dev, n);
    else kHistPrivate<<<n / T, T>>>(data.dev, bins.dev, n);
    ST_LAUNCHED();
    std::vector<unsigned> b = bins.host();
    for (int k = 0; k < 4; ++k) ST_REQUIRE(b[k] == want[k], "bin %d holds %u, expected %u", k, b[k], want[k]);
    return "";
}
static std::string tAtomicAdd() { return run(0); }
static std::string tAtomicRef() { return run(1); }
static std::string tPrivate() { return run(2); }

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"histogram with atomicAdd", tAtomicAdd},
    {"histogram with cuda::atomic_ref::fetch_add", tAtomicRef},
    {"histogram with a per-block shared copy merged atomically", tPrivate},
};
ST_END_CASES
