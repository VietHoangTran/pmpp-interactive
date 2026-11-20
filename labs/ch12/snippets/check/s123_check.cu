// Runs ../s123_fig12_3.cu (Fig. 12.3 as printed) or, built with -DSNIPPET_CG,
// ../s123_fig12_4_cg.cu (Fig. 12.4) on the course's CPU emulator for section 12.3, and
// prints what app/src/chapters/ch12/S123.snippets.test.ts compares with the model behind
// the section (app/src/sim/warpvote.ts). Under each of the emulator's three thread orders:
//   warp <name>: one warp, the listed lanes pass, the counter starts at 40; prints
//                lane>slot for each passing lane, then the counter afterwards
//   many:        4,000 keys in blocks of 128: is output[0, count) exactly the kept keys,
//                and do each warp's kept keys sit side by side, in lane order
// A passing key's value is its index, so the output says which thread wrote where; a key
// that fails is its index plus kFail. The threshold is raised to let every index pass.
#define FILTER_THRESHOLD 1000000u
#include <cstdio>
#include <algorithm>
#include <vector>
#if defined(SNIPPET_CG)
#include "../s123_fig12_4_cg.cu"
#else
#include "../s123_fig12_3.cu"
#endif

static const unsigned int kWarp = 32;  // the emulator's warp, and the book's: Fig. 12.3's masks are 32 bits
static const unsigned int kFail = 2000000u;

#if defined(PMPP_EMU)
static const int kOrders = 3;
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "in order" : i == 1 ? "reversed" : "shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

// Runs the kernel over `in` with the counter at `counter`; returns the counter afterwards.
static unsigned int run(const std::vector<unsigned int>& in, std::vector<unsigned int>& out, unsigned int counter,
                        unsigned int block) {
    const unsigned int n = (unsigned int)in.size();
    unsigned int *in_d, *out_d, *cnt_d;
    cudaMalloc((void**)&in_d, n * sizeof(unsigned int));
    cudaMalloc((void**)&out_d, out.size() * sizeof(unsigned int));
    cudaMalloc((void**)&cnt_d, sizeof(unsigned int));
    cudaMemcpy(in_d, in.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out.data(), out.size() * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(cnt_d, &counter, sizeof counter, cudaMemcpyHostToDevice);
    filter_kernel<<<(n + block - 1) / block, block>>>(in_d, out_d, n, cnt_d);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), out_d, out.size() * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&counter, cnt_d, sizeof counter, cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
    cudaFree(cnt_d);
    return counter;
}

// Returns 0 if output[0, count) holds exactly the kept keys; sets sideBySide.
static int many(bool& sideBySide) {
    const unsigned int n = 4000, block = 128;
    std::vector<unsigned int> in(n), out(n, 0xFFFFFFFFu), want;
    unsigned int s = 12345;
    for (unsigned int i = 0; i < n; ++i) {
        s = s * 1103515245u + 12345u;
        in[i] = ((s >> 16) % 3 == 0) ? kFail + i : i;   // about two keys in three pass
        if (in[i] < kFail) want.push_back(i);
    }
    unsigned int count = run(in, out, 0, block);
    std::vector<unsigned int> got(out.begin(), out.begin() + std::min(count, n));
    // Side by side: the keys of one warp form one run of the output, in lane order.
    sideBySide = true;
    std::vector<unsigned int> runs;
    for (size_t k = 0; k < got.size(); ++k) {
        if (k > 0 && got[k] / kWarp == got[k - 1] / kWarp) {
            if (got[k] < got[k - 1]) sideBySide = false;
        } else {
            runs.push_back(got[k] / kWarp);
        }
    }
    std::sort(runs.begin(), runs.end());
    if (std::adjacent_find(runs.begin(), runs.end()) != runs.end()) sideBySide = false;
    std::sort(got.begin(), got.end());
    return count == want.size() && got == want ? 0 : 1;
}

int main() {
    struct Case { const char* name; std::vector<unsigned int> pass; };
    std::vector<unsigned int> every;
    for (unsigned int l = 0; l < kWarp; ++l) every.push_back(l);
    const Case cases[] = {
        {"book", {1, 2, 4, 5}},
        {"rank", {2, 4, 5, 7}},
        {"all", every},
        {"lane31", {kWarp - 1}},
    };
    int failures = 0;
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        for (const Case& c : cases) {
            std::vector<unsigned int> in(kWarp), out(3 * kWarp, 0xFFFFFFFFu);
            for (unsigned int l = 0; l < kWarp; ++l)
                in[l] = std::find(c.pass.begin(), c.pass.end(), l) != c.pass.end() ? l : kFail + l;
            unsigned int after = run(in, out, 40, kWarp);
            std::printf("warp %s %s:", c.name, order);
            for (unsigned int lane : c.pass) {
                int slot = -1;
                for (unsigned int k = 0; k < out.size(); ++k)
                    if (out[k] == lane) slot = (int)k;
                std::printf(" %u>%d", lane, slot);
            }
            std::printf(" counter %u\n", after);
        }
        bool side = false;
        int bad = many(side);
        failures += bad;
        std::printf("many %s: %s, %s\n", order, bad ? "wrong" : "right", side ? "side by side" : "scattered");
    }
    std::printf("%s\n", failures ? "FAILED" : "DONE");
    return failures ? 1 : 0;
}
