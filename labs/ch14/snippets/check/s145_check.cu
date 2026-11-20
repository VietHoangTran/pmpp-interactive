// Runs ../s145_radix_iter.cu (section 14.5) on the course's CPU emulator and prints what
// app/src/chapters/ch14/S145.snippets.test.ts compares with app/src/sim/radixsort.ts.
// Under each of the emulator's three thread orders:
//   fig <bit>:   the book's sixteen keys in four blocks of four threads (Figs. 14.5, 14.6),
//                one pass on that bit, run kernel by kernel: the bits, the scan's N + 1
//                entries, each thread's destination, the output
//   passes:      four calls of radix_sort_iter on the book's keys, bit 0 to bit 3, with the
//                input and output swapped between calls: the list after each pass (Fig. 14.4)
//   keys <seed>: sixteen pseudo-random 8-bit keys, printed, then the destinations of a
//                pass on each of the eight bits (blocks of four)
//   big:         1,000 pseudo-random 32-bit keys, 32 passes in blocks of 256 threads (and a
//                scan of more entries than its 256 threads), against std::sort
// A thread's destination is found by tagging each key with its index above the bits the
// pass reads: output[dst] then says which thread wrote it.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "../s145_radix_iter.cu"

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

static const unsigned int kBook[16] = {12, 3, 6, 9, 15, 8, 5, 10, 9, 6, 11, 13, 4, 10, 7, 0};

static void print(const char* head, const std::vector<unsigned int>& v) {
    std::printf("%s", head);
    for (size_t k = 0; k < v.size(); ++k) std::printf(" %u", v[k]);
}

struct Dev {
    unsigned int *in, *out, *bits;
    unsigned int n;
    explicit Dev(unsigned int n_) : n(n_) {
        cudaMalloc((void**)&in, (n ? n : 1) * sizeof(unsigned int));
        cudaMalloc((void**)&out, (n ? n : 1) * sizeof(unsigned int));
        cudaMalloc((void**)&bits, (n + 1) * sizeof(unsigned int));   // N + 1 entries
    }
    ~Dev() {
        cudaFree(in);
        cudaFree(out);
        cudaFree(bits);
    }
    void put(const std::vector<unsigned int>& v) { cudaMemcpy(in, v.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice); }
    std::vector<unsigned int> get(unsigned int* p, unsigned int count) {
        std::vector<unsigned int> v(count);
        cudaMemcpy(v.data(), p, count * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        return v;
    }
};

// Destinations of one pass on `bit`, keys tagged with their index above bit `tagShift`.
static std::vector<unsigned int> destinations(const std::vector<unsigned int>& keys, unsigned int bit, unsigned int tagShift,
                                              unsigned int block) {
    const unsigned int n = (unsigned int)keys.size();
    Dev d(n);
    std::vector<unsigned int> tagged(n);
    for (unsigned int i = 0; i < n; ++i) tagged[i] = keys[i] | (i << tagShift);
    d.put(tagged);
    radix_sort_iter(d.in, d.out, d.bits, n, bit, block);
    std::vector<unsigned int> out = d.get(d.out, n), dst(n, 0xFFFFFFFFu);
    for (unsigned int j = 0; j < n; ++j) dst[out[j] >> tagShift] = j;
    return dst;
}

static void fig(unsigned int bit) {
    const unsigned int n = 16, block = 4;
    std::vector<unsigned int> keys(kBook, kBook + n), tagged(n);
    for (unsigned int i = 0; i < n; ++i) tagged[i] = keys[i] | (i << 8);
    Dev d(n);
    d.put(tagged);
    const unsigned int blocks = (n + block - 1) / block;
    radix_bits_kernel<<<blocks, block>>>(d.in, d.bits, n, bit);
    std::vector<unsigned int> bits = d.get(d.bits, n);
    exclusive_scan_kernel<<<1, SCAN_THREADS>>>(d.bits, n);
    std::vector<unsigned int> scan = d.get(d.bits, n + 1);
    radix_scatter_kernel<<<blocks, block>>>(d.in, d.out, d.bits, n, bit);
    std::vector<unsigned int> out = d.get(d.out, n), dst(n), plain(n);
    for (unsigned int j = 0; j < n; ++j) {
        dst[out[j] >> 8] = j;
        plain[j] = out[j] & 0xFFu;
    }
    std::printf("fig %u:", bit);
    print(" bits", bits);
    print(" | scan", scan);
    print(" | dst", dst);
    print(" | out", plain);
    std::printf("\n");
}

static void passes() {
    const unsigned int n = 16;
    Dev d(n);
    d.put(std::vector<unsigned int>(kBook, kBook + n));
    std::printf("passes:");
    for (unsigned int iter = 0; iter < 4; ++iter) {
        radix_sort_iter(d.in, d.out, d.bits, n, iter, 4);
        std::swap(d.in, d.out);
        print(iter ? " |" : "", d.get(d.in, n));
    }
    std::printf("\n");
}

static std::vector<unsigned int> lcg(unsigned int n, unsigned int seed, unsigned int mask) {
    std::vector<unsigned int> v(n);
    unsigned int s = seed;
    for (unsigned int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        v[i] = ((s >> 8) ^ (s << 13)) & mask;
    }
    return v;
}

static void randomKeys(unsigned int seed) {
    std::vector<unsigned int> keys = lcg(16, seed, 0xFFu);
    std::printf("keys %u:", seed);
    print("", keys);
    for (unsigned int bit = 0; bit < 8; ++bit) {
        std::printf(" | bit %u", bit);
        print("", destinations(keys, bit, 8, 4));
    }
    std::printf("\n");
}

static void big() {
    const unsigned int n = 1000;
    std::vector<unsigned int> keys = lcg(n, 77, 0xFFFFFFFFu), want = keys;
    std::sort(want.begin(), want.end());
    Dev d(n);
    d.put(keys);
    for (unsigned int iter = 0; iter < 32; ++iter) {
        radix_sort_iter(d.in, d.out, d.bits, n, iter, 256);
        std::swap(d.in, d.out);
    }
    std::printf("big: %s\n", d.get(d.in, n) == want ? "sorted" : "WRONG");
}

int main() {
    for (int o = 0; o < kOrders; ++o) {
        std::printf("order %s\n", useOrder(o));
        for (unsigned int bit = 0; bit < 4; ++bit) fig(bit);
        passes();
        randomKeys(5);
        randomKeys(1234);
        big();
    }
    std::printf("DONE\n");
    return 0;
}
