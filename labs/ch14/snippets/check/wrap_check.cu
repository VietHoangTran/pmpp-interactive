// Runs ../wrap_blocksort.cu (chapter 14's wrap page, question ch14/wrap/q-blocksort) on the
// course's CPU emulator and prints what app/src/chapters/ch14/Wrap.snippets.test.ts checks:
// the kernel as shown (no barrier between steps) against the corrected copy below, which adds
// one line. Under each of the emulator's three thread orders, for each case:
//   shown <case>: sorted yes|no, permutation yes|no
//   fixed <case>: sorted yes|no, permutation yes|no
//   fixed zero-one 8: <k> of 256 sorted       (every input of eight 0s and 1s)
// Only for the emu flavour, which runs one thread at a time between barriers: without a
// barrier in the loop, each thread there runs all its steps before the next one starts.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "../wrap_blocksort.cu"

// The corrected kernel: the snippet's, with __syncthreads() at the end of every step. In its
// own namespace so that the emulator's per-file kernel registration does not collide with the
// snippet's. The test checks that it differs from the snippet by that one line only.
namespace fixed {
__global__ void blockOddEvenSort(unsigned int* keys, unsigned int n) {
    __shared__ unsigned int s[MAXN];
    unsigned int t = threadIdx.x;
    for (unsigned int i = t; i < n; i += blockDim.x)
        s[i] = keys[i];
    __syncthreads();
    for (unsigned int step = 0; step < n; ++step) {
        unsigned int i = 2 * t + (step & 1);
        if (i + 1 < n && s[i] > s[i + 1]) {
            unsigned int tmp = s[i];
            s[i] = s[i + 1];
            s[i + 1] = tmp;
        }
        __syncthreads();
    }
    __syncthreads();
    for (unsigned int i = t; i < n; i += blockDim.x)
        keys[i] = s[i];
}
}  // namespace fixed

#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
static const int kOrders = 3;
// The same three orders, and the same shuffle seed, as labs/common/src/backend_cudalike.cpp.
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "in order" : i == 1 ? "reversed" : "shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

/** One launch of one block of (n + 1) / 2 threads; returns the keys afterwards. */
static std::vector<unsigned int> run(bool corrected, const std::vector<unsigned int>& v) {
    const unsigned int n = (unsigned int)v.size();
    unsigned int* d;
    cudaMalloc((void**)&d, n * sizeof(unsigned int));
    cudaMemcpy(d, v.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (corrected)
        fixed::blockOddEvenSort<<<1, (n + 1) / 2>>>(d, n);
    else
        blockOddEvenSort<<<1, (n + 1) / 2>>>(d, n);
    std::vector<unsigned int> out(n);
    cudaMemcpy(out.data(), d, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return out;
}

static void check(const char* which, bool corrected, const char* name, const std::vector<unsigned int>& v) {
    std::vector<unsigned int> got = run(corrected, v), a = got, b = v;
    std::sort(a.begin(), a.end());
    std::sort(b.begin(), b.end());
    std::printf("%s %s: sorted %s, permutation %s\n", which, name, std::is_sorted(got.begin(), got.end()) ? "yes" : "no",
                a == b ? "yes" : "no");
}

int main() {
    unsigned int seed = 2024;
    auto next = [&seed]() { seed = seed * 1103515245u + 12345u; return seed >> 16; };
    std::vector<unsigned int> rev64(64), rnd64(64), rnd129(129), rev256(256);
    for (unsigned int k = 0; k < 64; ++k) rev64[k] = 64 - k;
    for (auto& x : rnd64) x = next() % 1000;
    for (auto& x : rnd129) x = next() % 50;   // many ties, and an odd count
    for (unsigned int k = 0; k < 256; ++k) rev256[k] = 256 - k;
    for (int o = 0; o < kOrders; ++o) {
        std::printf("order %s\n", useOrder(o));
        for (int corrected = 0; corrected < 2; ++corrected) {
            const char* which = corrected ? "fixed" : "shown";
            check(which, corrected, "reversed 64", rev64);
            check(which, corrected, "random 64", rnd64);
            check(which, corrected, "random 129", rnd129);
            check(which, corrected, "reversed 256", rev256);
        }
        int ok = 0;
        for (unsigned int m = 0; m < 256; ++m) {
            std::vector<unsigned int> z(8);
            for (unsigned int k = 0; k < 8; ++k) z[k] = (m >> k) & 1;
            std::vector<unsigned int> got = run(true, z);
            ok += std::is_sorted(got.begin(), got.end()) ? 1 : 0;
        }
        std::printf("fixed zero-one 8: %d of 256 sorted\n", ok);
    }
    std::printf("DONE\n");
    return 0;
}
