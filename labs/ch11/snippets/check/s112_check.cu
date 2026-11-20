// Runs one kernel of ../s112_kogge_stone.cu or ../s112_race_variants.cu on one block and
// prints, per thread order, what the block wrote to output:
//
//   s112_check <kernel> v0 v1 ... v(n-1)        (one block of n threads, n <= SEG_SIZE)
//   -> <kernel> <order> <output[0]> ... <output[n-1]>     one line per order
//
// On the course's CPU emulator (PMPP_EMU) it runs the three orders the labs use; elsewhere
// it runs once. app/src/sim/ksblock.test.ts appends this file to the two snippets,
// translates the result with labs/tools/cu2emu.py, builds it as the lab runner builds the
// emu backend (AddressSanitizer on, SEG_SIZE 16), and compares each line with the model in
// app/src/sim/ksblock.ts. The emulator's "forward" order is the model's "each thread runs
// as far as it can, thread 0 first"; "reverse" is the same from the last thread.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef SEG_SIZE
#define SEG_SIZE 1024
#endif

__global__ void kogge_stone_scan_kernel(float *input, float *output, unsigned int N);
__global__ void double_buffer_scan_kernel(float* input, float* output, unsigned int N);
__global__ void no_barrier2_scan_kernel(float *input, float *output, unsigned int N);
__global__ void no_barrier1_scan_kernel(float *input, float *output, unsigned int N);
__global__ void direct_add_scan_kernel(float *input, float *output, unsigned int N);
__global__ void no_copy_double_buffer_scan_kernel(float* input, float* output, unsigned int N);

typedef void (*Kernel)(float*, float*, unsigned int);
struct Entry {
    const char* name;
    Kernel kernel;
};
static const Entry kKernels[] = {
    {"kogge_stone_scan_kernel", kogge_stone_scan_kernel},
    {"double_buffer_scan_kernel", double_buffer_scan_kernel},
    {"no_barrier2_scan_kernel", no_barrier2_scan_kernel},
    {"no_barrier1_scan_kernel", no_barrier1_scan_kernel},
    {"direct_add_scan_kernel", direct_add_scan_kernel},
    {"no_copy_double_buffer_scan_kernel", no_copy_double_buffer_scan_kernel},
};

#if defined(PMPP_EMU)
static const int kOrders = 3;
// The same three orders, and the same shuffle seed, as labs/common/src/backend_cudalike.cpp.
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "forward" : i == 1 ? "reverse" : "shuffle";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <kernel> v0 v1 ...\n", argv[0]);
        return 2;
    }
    Kernel kernel = nullptr;
    for (const Entry& e : kKernels)
        if (std::strcmp(e.name, argv[1]) == 0) kernel = e.kernel;
    if (!kernel) {
        std::fprintf(stderr, "no kernel named %s\n", argv[1]);
        return 2;
    }
    const unsigned n = unsigned(argc - 2);
    if (n > SEG_SIZE) {
        std::fprintf(stderr, "at most %d values\n", SEG_SIZE);
        return 2;
    }
    std::vector<float> values(n);
    for (unsigned k = 0; k < n; ++k) values[k] = (float)std::atof(argv[2 + k]);

    float* input = nullptr;
    float* output = nullptr;
    cudaMalloc((void**)&input, n * sizeof(float));
    cudaMalloc((void**)&output, n * sizeof(float));
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<float> unset(n, NAN);
        cudaMemcpy(input, values.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(output, unset.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        kernel<<<1, n>>>(input, output, n);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("%s %s launch-failed\n", argv[1], order);
            continue;
        }
        std::vector<float> after(n);
        cudaMemcpy(after.data(), output, n * sizeof(float), cudaMemcpyDeviceToHost);
        std::printf("%s %s", argv[1], order);
        for (unsigned k = 0; k < n; ++k) std::printf(" %.9g", after[k]);
        std::printf("\n");
    }
    cudaFree(input);
    cudaFree(output);
    return 0;
}
