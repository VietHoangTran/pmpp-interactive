// Runs ../s144_floatkey.cu (section 14.4) on the course's CPU emulator and prints what
// app/src/chapters/ch14/S144.snippets.test.ts compares, bit for bit, with
// app/src/sim/floatkey.ts. Under each of the emulator's three thread orders:
//   key <in> <asc> <desc>:  one line per input bit pattern: E1's ten values, two quiet
//                           NaNs, then 600 patterns from a fixed pseudo-random sequence
//   sort <rule>:            the twelve E1-and-NaN values sorted descending by stable_sort
//                           on the descending keys, printed as indices
// The input goes to the device as unsigned ints: the kernel never sees a float.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "../s144_floatkey.cu"

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

static std::vector<unsigned int> keysOnDevice(const std::vector<unsigned int>& in, bool descending) {
    const unsigned int n = (unsigned int)in.size();
    unsigned int *in_d, *out_d;
    cudaMalloc((void**)&in_d, n * sizeof(unsigned int));
    cudaMalloc((void**)&out_d, n * sizeof(unsigned int));
    cudaMemcpy(in_d, in.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    const unsigned int block = 128;
    logits_to_keys<<<(n + block - 1) / block, block>>>(in_d, out_d, n, descending);
    std::vector<unsigned int> out(n);
    cudaMemcpy(out.data(), out_d, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
    return out;
}

int main() {
    // E1 (notes/today/ch14-facts.md): -inf, -FLT_MAX, -1.0, smallest negative subnormal,
    // -0.0, +0.0, smallest positive subnormal, 1.0, FLT_MAX, +inf; then +qNaN and -qNaN.
    std::vector<unsigned int> in = {0xFF800000u, 0xFF7FFFFFu, 0xBF800000u, 0x80000001u, 0x80000000u, 0x00000000u,
                                    0x00000001u, 0x3F800000u, 0x7F7FFFFFu, 0x7F800000u, 0x7FC00000u, 0xFFC00000u};
    const size_t edges = in.size();
    unsigned int s = 2026;
    for (int k = 0; k < 600; ++k) {
        s = s * 1103515245u + 12345u;
        unsigned int t = s * 2654435761u;
        in.push_back((s & 0xFFFF0000u) | (t >> 16));
    }
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<unsigned int> asc = keysOnDevice(in, false), desc = keysOnDevice(in, true);
        std::printf("order %s\n", order);
        for (size_t k = 0; k < in.size(); ++k) std::printf("key %08X %08X %08X\n", in[k], asc[k], desc[k]);
        std::vector<unsigned int> idx(edges);
        for (size_t k = 0; k < edges; ++k) idx[k] = (unsigned int)k;
        std::stable_sort(idx.begin(), idx.end(), [&](unsigned int a, unsigned int b) { return desc[a] < desc[b]; });
        std::printf("sort desc:");
        for (unsigned int k : idx) std::printf(" %u", k);
        std::printf("\n");
    }
    std::printf("DONE\n");
    return 0;
}
