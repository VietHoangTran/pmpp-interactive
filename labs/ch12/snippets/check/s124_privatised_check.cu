// Runs one kernel of ../s124_privatised_variants.cu on four blocks of four threads and
// prints the public counter and the whole output buffer:
//
//   s124_privatised_check <kernel> <order> f0 f1 ... f15     (each f is a keep flag, 0 or 1)
//   -> <kernel> <order> <outputSize> <output[0]> ... <output[15]>
//
// <order> is forward, reverse or shuffle on the course's emulator (the same three orders,
// and the same shuffle seed, as the labs use), and anything on the race-detection flavour,
// which runs once. Key i is stored as 2i + f_i, so cond() (odd values) keeps exactly the
// keys whose flag is 1. The output buffer starts as 0xFF bytes, so a slot nobody wrote
// prints as 4294967295. One order per process: a kernel that writes out of range stops the
// process with an AddressSanitizer report, and the others must still run.
// app/src/sim/privfilter.test.ts appends this file to s124_privatised_variants.cu,
// translates the result with labs/tools/cu2emu.py, builds it as the lab runner builds the
// emu and race backends, and compares every run with app/src/sim/privfilter.ts.
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void filterPrivatised(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize);
__global__ void filterNoBarrier1(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize);
__global__ void filterNoBarrier2(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize);
__global__ void filterNoBarrier3(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize);
__global__ void filterPlainCounter(const unsigned int* input, unsigned int* output, unsigned int N, unsigned int* outputSize);

typedef void (*Kernel)(const unsigned int*, unsigned int*, unsigned int, unsigned int*);
struct Entry {
    const char* name;
    Kernel kernel;
};
static const Entry kKernels[] = {
    {"filterPrivatised", filterPrivatised},
    {"filterNoBarrier1", filterNoBarrier1},
    {"filterNoBarrier2", filterNoBarrier2},
    {"filterNoBarrier3", filterNoBarrier3},
    {"filterPlainCounter", filterPlainCounter},
};

static const unsigned int kBlocks = 4;
static const unsigned int kThreads = 4;

int main(int argc, char** argv) {
    const unsigned int n = kBlocks * kThreads;
    if (argc != 3 + (int)n) {
        std::fprintf(stderr, "usage: %s <kernel> <forward|reverse|shuffle> f0 ... f%u\n", argv[0], n - 1);
        return 2;
    }
    Kernel kernel = nullptr;
    for (const Entry& e : kKernels)
        if (std::strcmp(e.name, argv[1]) == 0) kernel = e.kernel;
    if (!kernel) {
        std::fprintf(stderr, "no kernel named %s\n", argv[1]);
        return 2;
    }
    const char* order = argv[2];
#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
    // The labs' seed (labs/common/src/backend_cudalike.cpp).
    using pmpp_emu::Schedule;
    const Schedule s = std::strcmp(order, "reverse") == 0 ? Schedule::Reverse
                       : std::strcmp(order, "shuffle") == 0 ? Schedule::Shuffle
                                                            : Schedule::Forward;
    pmpp_emu::set_schedule(s, 7);
#endif
    std::vector<unsigned int> input(n);
    for (unsigned int k = 0; k < n; ++k) input[k] = 2u * k + (std::atoi(argv[3 + k]) ? 1u : 0u);

    unsigned int* input_d = nullptr;
    unsigned int* output_d = nullptr;
    unsigned int* size_d = nullptr;
    cudaMalloc((void**)&input_d, n * sizeof(unsigned int));
    cudaMalloc((void**)&output_d, n * sizeof(unsigned int));
    cudaMalloc((void**)&size_d, sizeof(unsigned int));
    cudaMemcpy(input_d, input.data(), n * sizeof(unsigned int), cudaMemcpyHostToDevice);
    cudaMemset(output_d, 0xFF, n * sizeof(unsigned int));
    cudaMemset(size_d, 0, sizeof(unsigned int));
    kernel<<<kBlocks, kThreads>>>(input_d, output_d, n, size_d);
    if (cudaDeviceSynchronize() != cudaSuccess) {
        std::printf("%s %s launch-failed\n", argv[1], order);
        return 3;
    }
    std::vector<unsigned int> output(n);
    unsigned int size = 0;
    cudaMemcpy(output.data(), output_d, n * sizeof(unsigned int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&size, size_d, sizeof(unsigned int), cudaMemcpyDeviceToHost);
    std::printf("%s %s %u", argv[1], order, size);
    for (unsigned int k = 0; k < n; ++k) std::printf(" %u", output[k]);
    std::printf("\n");
    cudaFree(input_d);
    cudaFree(output_d);
    cudaFree(size_d);
    return 0;
}
