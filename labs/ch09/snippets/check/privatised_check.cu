// Runs one kernel of ../privatised_variants.cu on two blocks of four threads and prints,
// per thread order, the NUM_BINS public bins it produced:
//
//   privatised_check <kernel> v0 v1 ... v7        (each v is a bin number, 0 to 7)
//   -> <kernel> <order> <bins[0]> ... <bins[7]>   one line per order
//
// On the course's emulator (PMPP_EMU) it runs the three orders the labs use, with the
// same shuffle seed; in the race-detection flavour, and anywhere else, it runs once.
// app/src/sim/histpriv.test.ts appends this file to privatised_variants.cu, translates
// the result with labs/tools/cu2emu.py, builds it as the lab runner builds the emu and
// race backends, runs every kernel, and compares the bins (emu) and the ThreadSanitizer
// reports (race) with the model in app/src/sim/histpriv.ts.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void histPrivatised(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histNoZeroing(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histNoBarrierAfterZeroing(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histNoBarrierBeforeCommit(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histPlainSharedUpdate(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histPlainCommit(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histBlockScopeCommit(const unsigned char* image, unsigned int* bins, unsigned int n);
__global__ void histCommitBoundBlockDim(const unsigned char* image, unsigned int* bins, unsigned int n);

typedef void (*Kernel)(const unsigned char*, unsigned int*, unsigned int);
struct Entry {
    const char* name;
    Kernel kernel;
};
static const Entry kKernels[] = {
    {"histPrivatised", histPrivatised},
    {"histNoZeroing", histNoZeroing},
    {"histNoBarrierAfterZeroing", histNoBarrierAfterZeroing},
    {"histNoBarrierBeforeCommit", histNoBarrierBeforeCommit},
    {"histPlainSharedUpdate", histPlainSharedUpdate},
    {"histPlainCommit", histPlainCommit},
    {"histBlockScopeCommit", histBlockScopeCommit},
    {"histCommitBoundBlockDim", histCommitBoundBlockDim},
};

static const unsigned int kBins = 8;
static const unsigned int kBlocks = 2;
static const unsigned int kThreads = 4;

#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
static const int kOrders = 3;
// The same three orders, and the same shuffle seed, as labs/common/src/backend_cudalike.cpp.
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "forward" : i == 1 ? "reverse" : "shuffle";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "once"; }
#endif

int main(int argc, char** argv) {
    const unsigned int n = kBlocks * kThreads;
    if (argc != 2 + (int)n) {
        std::fprintf(stderr, "usage: %s <kernel> v0 ... v%u\n", argv[0], n - 1);
        return 2;
    }
    Kernel kernel = nullptr;
    for (const Entry& e : kKernels)
        if (std::strcmp(e.name, argv[1]) == 0) kernel = e.kernel;
    if (!kernel) {
        std::fprintf(stderr, "no kernel named %s\n", argv[1]);
        return 2;
    }
    std::vector<unsigned char> image(n);
    for (unsigned int k = 0; k < n; ++k) image[k] = (unsigned char)(std::atoi(argv[2 + k]) % kBins);

    unsigned char* image_d = nullptr;
    unsigned int* bins_d = nullptr;
    cudaMalloc((void**)&image_d, n);
    cudaMalloc((void**)&bins_d, kBins * sizeof(unsigned int));
    cudaMemcpy(image_d, image.data(), n, cudaMemcpyHostToDevice);
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        cudaMemset(bins_d, 0, kBins * sizeof(unsigned int));
        kernel<<<kBlocks, kThreads>>>(image_d, bins_d, n);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("%s %s launch-failed\n", argv[1], order);
            continue;
        }
        std::vector<unsigned int> bins(kBins);
        cudaMemcpy(bins.data(), bins_d, kBins * sizeof(unsigned int), cudaMemcpyDeviceToHost);
        std::printf("%s %s", argv[1], order);
        for (unsigned int b = 0; b < kBins; ++b) std::printf(" %u", bins[b]);
        std::printf("\n");
    }
    cudaFree(image_d);
    cudaFree(bins_d);
    return 0;
}
