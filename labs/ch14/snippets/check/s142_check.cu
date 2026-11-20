// Runs ../s142_oddeven.cu (section 14.2) on the course's CPU emulator and prints what
// app/src/chapters/ch14/S142.snippets.test.ts compares with the model behind the section
// (app/src/sim/oddeven.ts). Under each of the emulator's three thread orders:
//   rows <name>:  single launches of sort_kernel, even step first; the list after each
//                 step, then the flag each step left
//   sort <name>:  oddEvenSort, the host loop: steps launched, then the list it returns
//   random:       520 keys (two blocks of 256 threads per step) against std::sort
//   empty:        N = 0 launches nothing and the kernel touches nothing
// Built with -DSNIPPET_AS_PRINTED it instead runs one step of Fig. 14.2 exactly as printed
// (plain store to the flag) on 64 reversed keys, for the race-detection emulator; without
// it, the same step with the snippet's atomic store.
#include <algorithm>
#include <cstdio>
#include <vector>
#include "../s142_oddeven.cu"

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

static void print(const char* head, const std::vector<unsigned int>& v) {
    std::printf("%s", head);
    for (size_t k = 0; k < v.size(); ++k) std::printf("%s%u", k ? " " : "", v[k]);
}

// Single launches, as the stepper shows them.
static void rows(const char* name, std::vector<unsigned int> v, unsigned int steps) {
    const unsigned int N = (unsigned int)v.size();
    unsigned int *d, *flag;
    cudaMalloc((void**)&d, N * sizeof(unsigned int));
    cudaMalloc((void**)&flag, sizeof(unsigned int));
    cudaMemcpy(d, v.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    std::vector<unsigned int> flags;
    std::printf("rows %s:", name);
    for (unsigned int s = 0; s < steps; ++s) {
        unsigned int f = 0;
        cudaMemcpy(flag, &f, sizeof f, cudaMemcpyHostToDevice);
        sort_kernel<<<1, N / 2>>>(d, flag, N, s % 2);
        cudaMemcpy(v.data(), d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        cudaMemcpy(&f, flag, sizeof f, cudaMemcpyDeviceToHost);
        flags.push_back(f);
        print(s ? " | " : " ", v);
    }
    print(" | flags ", flags);
    std::printf("\n");
    cudaFree(d);
    cudaFree(flag);
}

static std::vector<unsigned int> sortOnDevice(std::vector<unsigned int> v, unsigned int& steps) {
    const unsigned int N = (unsigned int)v.size();
    unsigned int* d;
    cudaMalloc((void**)&d, (N ? N : 1) * sizeof(unsigned int));
    if (N) cudaMemcpy(d, v.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    steps = oddEvenSort(d, N);
    if (N) cudaMemcpy(v.data(), d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaFree(d);
    return v;
}

static void sortCase(const char* name, const std::vector<unsigned int>& v) {
    unsigned int steps = 0;
    std::vector<unsigned int> out = sortOnDevice(v, steps);
    std::printf("sort %s: steps %u out", name, steps);
    print(" ", out);
    std::printf("\n");
}

#if defined(SNIPPET_AS_PRINTED)
// Fig. 14.2 exactly as printed (p. 332). In its own namespace so that the emulator's
// per-file kernel registration does not collide with the snippet's.
namespace printed {
__global__ void sort_kernel(unsigned int* data, unsigned int* hasChanged,
                            unsigned int N, unsigned int isOddStep) {
    unsigned int i = 2*(blockIdx.x*blockDim.x + threadIdx.x)
                     + (isOddStep ? 1 : 0);
    if(i < N - 1) {
        if(data[i] > data[i + 1]) {
            unsigned int tmp = data[i];
            data[i] = data[i + 1];
            data[i + 1] = tmp;
            *hasChanged = 1;
        }
    }
}
}  // namespace printed
#endif

[[maybe_unused]] static int raceStep() {
    const unsigned int N = 64;
    std::vector<unsigned int> v(N);
    for (unsigned int k = 0; k < N; ++k) v[k] = N - k;   // reversed: every pair of the even step swaps
    unsigned int *d, *flag, f = 0;
    cudaMalloc((void**)&d, N * sizeof(unsigned int));
    cudaMalloc((void**)&flag, sizeof(unsigned int));
    cudaMemcpy(d, v.data(), N * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemcpy(flag, &f, sizeof f, cudaMemcpyHostToDevice);
#if defined(SNIPPET_AS_PRINTED)
    printed::sort_kernel<<<1, N / 2>>>(d, flag, N, 0);
    const char* which = "as printed";
#else
    sort_kernel<<<1, N / 2>>>(d, flag, N, 0);
    const char* which = "atomic";
#endif
    cudaMemcpy(&f, flag, sizeof f, cudaMemcpyDeviceToHost);
    cudaMemcpy(v.data(), d, N * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    bool pairs = true;
    for (unsigned int k = 0; k < N; k += 2) pairs = pairs && v[k] < v[k + 1];
    std::printf("race step %s: flag %u, pairs ordered %s\n", which, f, pairs ? "yes" : "no");
    cudaFree(d);
    cudaFree(flag);
    return 0;
}

int main() {
#if defined(PMPP_EMU_RACE) || defined(SNIPPET_AS_PRINTED)
    raceStep();
    std::printf("DONE\n");
    return 0;
#else
    for (int o = 0; o < kOrders; ++o) {
        std::printf("order %s\n", useOrder(o));
        rows("fig", {4, 7, 2, 3, 8, 5, 9, 6}, 6);
        rows("ten", {9, 8, 7, 6, 5, 4, 3, 2, 1, 0}, 2);
        sortCase("fig", {4, 7, 2, 3, 8, 5, 9, 6});
        sortCase("largest", {8, 1, 2, 3, 4, 5, 6, 7});
        sortCase("trap", {1, 3, 2, 4, 5, 6, 7, 8});
        sortCase("trap4", {1, 3, 2, 4});
        sortCase("reversed", {9, 8, 7, 6, 5, 4, 3, 2});
        sortCase("odd7", {3, 2, 1, 0, 5, 4, 6});
        sortCase("one", {5});
        sortCase("empty", {});
        // Random keys in one block, then 520 keys (two blocks of 256 threads per step) that
        // are sorted except for short reversed runs, so that few steps are needed.
        unsigned int s = 12345;
        auto next = [&s]() { s = s * 1103515245u + 12345u; return s >> 16; };
        std::vector<unsigned int> r(40);
        for (auto& x : r) x = next() % 100;
        unsigned int steps = 0;
        std::vector<unsigned int> got = sortOnDevice(r, steps), want = r;
        std::sort(want.begin(), want.end());
        std::printf("random 40: %s\n", got == want ? "matches std::sort" : "WRONG");
        std::vector<unsigned int> w(520);
        for (unsigned int k = 0; k < 520; ++k) w[k] = k;
        for (unsigned int k = 0; k + 4 <= 520; k += 4 + next() % 9) std::reverse(w.begin() + k, w.begin() + k + 4);
        got = sortOnDevice(w, steps);
        want = w;
        std::sort(want.begin(), want.end());
        std::printf("two blocks 520: %s, steps %u\n", got == want ? "matches std::sort" : "WRONG", steps);
        // N == 0 straight to the kernel: one thread, a one-element buffer it must not touch.
        unsigned int *d, *flag, f = 0, sentinel = 77;
        cudaMalloc((void**)&d, sizeof(unsigned int));
        cudaMalloc((void**)&flag, sizeof(unsigned int));
        cudaMemcpy(d, &sentinel, sizeof sentinel, cudaMemcpyHostToDevice);
        cudaMemcpy(flag, &f, sizeof f, cudaMemcpyHostToDevice);
        sort_kernel<<<1, 1>>>(d, flag, 0, 0);
        cudaMemcpy(&sentinel, d, sizeof sentinel, cudaMemcpyDeviceToHost);
        cudaMemcpy(&f, flag, sizeof f, cudaMemcpyDeviceToHost);
        std::printf("kernel N=0: %s\n", sentinel == 77 && f == 0 ? "untouched" : "TOUCHED");
        cudaFree(d);
        cudaFree(flag);
    }
    std::printf("DONE\n");
    return 0;
#endif
}
