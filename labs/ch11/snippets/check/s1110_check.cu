// Runs the kernel of ../s1110_brent_kung.cu on N values, ceil(N / SEG_SIZE) blocks of
// SEG_SIZE/2 threads, and prints, per thread order, what it wrote to output:
//
//   s1110_check <kernel> v0 v1 ... v(n-1)
//   -> <kernel> <order> <output[0]> ... <output[n-1]>     one line per order
//
// <kernel> is brent_kung (the lesson's kernel) or, when built with S1110_WITH_VARIANT,
// no_barrier: the same kernel with the barrier inside both loops removed, which
// app/src/sim/prefixnet.test.ts generates from the snippet's own text as a negative
// control. That test appends this file to the snippet, translates the result with
// labs/tools/cu2emu.py, builds it as the lab runner builds the emu backend (AddressSanitizer
// on) and the race backend (ThreadSanitizer on), and compares every line with the model in
// app/src/sim/prefixnet.ts. On the emulator it runs the three thread orders the labs use;
// elsewhere it runs once.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#ifndef SEG_SIZE
#define SEG_SIZE 2048
#endif

__global__ void brent_kung_scan_kernel(float *input, float *output, unsigned int N);
#if defined(S1110_WITH_VARIANT)
__global__ void brent_kung_no_barrier_kernel(float *input, float *output, unsigned int N);
#endif

typedef void (*Kernel)(float*, float*, unsigned int);

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
    if (std::strcmp(argv[1], "brent_kung") == 0) kernel = brent_kung_scan_kernel;
#if defined(S1110_WITH_VARIANT)
    if (std::strcmp(argv[1], "no_barrier") == 0) kernel = brent_kung_no_barrier_kernel;
#endif
    if (!kernel) {
        std::fprintf(stderr, "no kernel named %s\n", argv[1]);
        return 2;
    }
    const unsigned int n = (unsigned int)(argc - 2);
    std::vector<float> values(n);
    for (unsigned int k = 0; k < n; ++k) values[k] = (float)std::atof(argv[2 + k]);
    const unsigned int blocks = (n + SEG_SIZE - 1) / SEG_SIZE;

    float* input = nullptr;
    float* output = nullptr;
    cudaMalloc((void**)&input, n * sizeof(float));
    cudaMalloc((void**)&output, n * sizeof(float));
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<float> unset(n, NAN);
        cudaMemcpy(input, values.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(output, unset.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        kernel<<<blocks, SEG_SIZE / 2>>>(input, output, n);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("%s %s launch-failed\n", argv[1], order);
            continue;
        }
        std::vector<float> got(n);
        cudaMemcpy(got.data(), output, n * sizeof(float), cudaMemcpyDeviceToHost);
        std::printf("%s %s", argv[1], order);
        for (unsigned int k = 0; k < n; ++k) std::printf(" %.9g", got[k]);
        std::printf("\n");
    }
    cudaFree(input);
    cudaFree(output);
    return 0;
}
