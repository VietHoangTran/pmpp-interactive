// Runs one kernel of ../s103_inplace.cu, ../s104_mirrored.cu or ../s106_barrier_hunt.cu on
// one block and prints, per thread order, what *output and the input hold afterwards:
//
//   reduction_check <kernel> <threads> v0 v1 ... v(n-1)
//   -> <kernel> <order> <output> <input[0]> ... <input[n-1]>     one line per order
//
// On the course's CPU emulator (PMPP_EMU) it runs the three orders the labs use; elsewhere
// it runs once. app/src/sim/reduction.test.ts appends this file to the three snippets,
// translates the result with labs/tools/cu2emu.py, builds it as the lab runner builds the
// emu backend (AddressSanitizer on), runs every kernel, and compares each line with the
// model in app/src/sim/reduction.ts. One kernel per process, because AddressSanitizer stops
// the process at the first out-of-bounds read.
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

__global__ void SimpleSumReductionKernel(float* input, float* output);
__global__ void ConvergentSumReductionKernel(float* input, float* output);
__global__ void MirroredSumReductionKernel(float* input, float* output);
__global__ void SharedMemorySumReductionKernel(float* input, float* output);
__global__ void BarrierAtBottomKernel(float* input, float* output);
__global__ void NoBarrierKernel(float* input, float* output);
__global__ void BarrierInsideIfKernel(float* input, float* output);
__global__ void LessOrEqualKernel(float* input, float* output);
__global__ void MaxThenSumKernel(float* input, float* output);
__global__ void MaxThenSumNoBarrierKernel(float* input, float* output);

typedef void (*Kernel)(float*, float*);
struct Entry {
    const char* name;
    Kernel kernel;
};
static const Entry kKernels[] = {
    {"SimpleSumReductionKernel", SimpleSumReductionKernel},
    {"ConvergentSumReductionKernel", ConvergentSumReductionKernel},
    {"MirroredSumReductionKernel", MirroredSumReductionKernel},
    {"SharedMemorySumReductionKernel", SharedMemorySumReductionKernel},
    {"BarrierAtBottomKernel", BarrierAtBottomKernel},
    {"NoBarrierKernel", NoBarrierKernel},
    {"BarrierInsideIfKernel", BarrierInsideIfKernel},
    {"LessOrEqualKernel", LessOrEqualKernel},
    {"MaxThenSumKernel", MaxThenSumKernel},
    {"MaxThenSumNoBarrierKernel", MaxThenSumNoBarrierKernel},
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
    if (argc < 4) {
        std::fprintf(stderr, "usage: %s <kernel> <threads> v0 v1 ...\n", argv[0]);
        return 2;
    }
    Kernel kernel = nullptr;
    for (const Entry& e : kKernels)
        if (std::strcmp(e.name, argv[1]) == 0) kernel = e.kernel;
    if (!kernel) {
        std::fprintf(stderr, "no kernel named %s\n", argv[1]);
        return 2;
    }
    const int threads = std::atoi(argv[2]);
    const int n = argc - 3;
    std::vector<float> values(n);
    for (int k = 0; k < n; ++k) values[k] = (float)std::atof(argv[3 + k]);

    float* input = nullptr;
    float* output = nullptr;
    cudaMalloc((void**)&input, n * sizeof(float));
    cudaMalloc((void**)&output, sizeof(float));
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        const float unset = NAN;
        cudaMemcpy(input, values.data(), n * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(output, &unset, sizeof(float), cudaMemcpyHostToDevice);
        kernel<<<1, threads>>>(input, output);
        if (cudaDeviceSynchronize() != cudaSuccess) {
            std::printf("%s %s launch-failed\n", argv[1], order);
            continue;
        }
        float result = 0;
        std::vector<float> after(n);
        cudaMemcpy(&result, output, sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(after.data(), input, n * sizeof(float), cudaMemcpyDeviceToHost);
        std::printf("%s %s %.9g", argv[1], order, result);
        for (int k = 0; k < n; ++k) std::printf(" %.9g", after[k]);
        std::printf("\n");
    }
    cudaFree(input);
    cudaFree(output);
    return 0;
}
