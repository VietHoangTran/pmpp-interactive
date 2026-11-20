// Self-test, negative control for the race detector: a histogram whose bins are
// updated with a plain increment. Many threads read the same old count and
// write back old + 1, so counts are lost on a GPU.
//   emu:  PASSES. One emulated thread runs at a time and is never interrupted
//         inside `bins[b]++`, so the baton emulator cannot see this bug.
//   race: must be REPORTED by ThreadSanitizer, with this file and line.
#include "selftest_cu.h"

__global__ void kHistPlain(const unsigned char* data, unsigned* bins, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) bins[data[i] / 64]++;            // BUG: read-modify-write without an atomic
}

static std::string tHistPlain() {
    const int n = 4096, T = 64;
    std::vector<unsigned char> h(n);
    unsigned want[4] = {0, 0, 0, 0};
    for (int i = 0; i < n; ++i) { h[i] = (unsigned char)((i * 131 + 7) % 256); want[h[i] / 64]++; }
    StBuf<unsigned char> data(h); StBuf<unsigned> bins(4, 0u);
    kHistPlain<<<n / T, T>>>(data.dev, bins.dev, n);
    ST_LAUNCHED();
    std::vector<unsigned> b = bins.host();
    for (int k = 0; k < 4; ++k) ST_REQUIRE(b[k] == want[k], "bin %d holds %u, expected %u (updates were lost)", k, b[k], want[k]);
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"histogram with plain increments", tHistPlain},
};
ST_END_CASES
