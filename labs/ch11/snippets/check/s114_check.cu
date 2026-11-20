// Runs the book's listings in ../s114_blockscan_as_printed.cu, ../s116_coarsened_as_printed.cu
// and ../s117_regtiled_as_printed.cu on the course's CPU emulator, for sections 11.4, 11.6
// and 11.7, and prints what they produce so that app/src/sim/warpscan.test.ts can compare
// it with the models behind those sections (app/src/sim/warpscan.ts, coarsescan.ts).
//
// The test does not include the three files: all three define a kernel called scan_kernel,
// so it renames them (scan_kernel_fig_11_10, _11_12, _11_13), puts the three files and this
// one into a single source, translates it with labs/tools/cu2emu.py and builds it with
// -DBLOCK_DIM and -DCOARSE_FACTOR. The declarations below let this file pass the CUDA
// front-end check on its own.
//
// Printed, in each of the emulator's three thread orders:
//   lanes <variant> <order>: the 64 results of a two-warp block, thread t passing
//       (5t mod 7) + 1. Variants: printed (Fig. 11.8), threadIdx (the test written with
//       threadIdx.x, written for this check), none (no test at all, written for this check)
//   fig11.10 / fig11.12 / fig11.13 <order>: <wrong> of <n>, each block's segment against
//       an inclusive scan of that segment computed on the host
#include <cstdio>
#include <vector>

#ifndef WARP_SIZE
#define WARP_SIZE 32
#endif
#ifndef BLOCK_DIM
#define BLOCK_DIM 1024
#endif
#ifndef COARSE_FACTOR
#define COARSE_FACTOR 4
#endif

__device__ float warpScan(float val);  // Fig. 11.8
__global__ void scan_kernel_fig_11_10(float* input, float* output, unsigned int N);
__global__ void scan_kernel_fig_11_12(float* input, float* output, unsigned int N);
__global__ void scan_kernel_fig_11_13(float* input, float* output, unsigned int N);

// Fig. 11.8 with its participation test written with threadIdx.x (the book warns against it).
__device__ float warpScanThreadIdx(float val) {
    for (unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
        float leftVal = __shfl_up_sync(0xffffffff, val, stride);
        if (threadIdx.x >= stride) {
            val += leftVal;
        }
    }
    return val;
}

// Fig. 11.8 with no participation test.
__device__ float warpScanNoTest(float val) {
    for (unsigned int stride = 1; stride < WARP_SIZE; stride *= 2) {
        float leftVal = __shfl_up_sync(0xffffffff, val, stride);
        val += leftVal;
    }
    return val;
}

__global__ void scanLanes(const float* in, float* out, int variant) {
    float v = in[threadIdx.x];
    out[threadIdx.x] = variant == 0 ? warpScan(v) : variant == 1 ? warpScanThreadIdx(v) : warpScanNoTest(v);
}

#if defined(PMPP_EMU)
static const int kOrders = 3;
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "in-order" : i == 1 ? "reversed" : "shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

static float value(unsigned int t) { return (float)((5u * t) % 7u + 1u); }

// Wrong elements when each segment of `seg` elements is compared with its own inclusive scan.
static int wrongSegments(const std::vector<float>& in, const std::vector<float>& out, unsigned int seg) {
    int wrong = 0;
    float run = 0.0f;
    for (size_t i = 0; i < in.size(); ++i) {
        run = (i % seg == 0) ? in[i] : run + in[i];
        wrong += out[i] != run;
    }
    return wrong;
}

static int runScan(void (*kernel)(float*, float*, unsigned int), const char* name, const char* order,
                   unsigned int n, unsigned int blocks, unsigned int seg) {
    std::vector<float> in(n), out(n, -1.0f);
    for (unsigned int i = 0; i < n; ++i) in[i] = value(i);
    float *in_d = nullptr, *out_d = nullptr;
    cudaMalloc((void**)&in_d, n * sizeof(float));
    cudaMalloc((void**)&out_d, n * sizeof(float));
    cudaMemcpy(in_d, in.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(out_d, out.data(), n * sizeof(float), cudaMemcpyHostToDevice);
    kernel<<<blocks, BLOCK_DIM>>>(in_d, out_d, n);
    cudaDeviceSynchronize();
    cudaMemcpy(out.data(), out_d, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(in_d);
    cudaFree(out_d);
    int wrong = wrongSegments(in, out, seg);
    std::printf("%s %s: %d of %u\n", name, order, wrong, n);
    return wrong;
}

int main() {
    const unsigned int lanes = 2 * WARP_SIZE;
    float in[2 * WARP_SIZE], out[2 * WARP_SIZE];
    for (unsigned int t = 0; t < lanes; ++t) in[t] = value(t);
    float *in_d = nullptr, *out_d = nullptr;
    cudaMalloc((void**)&in_d, sizeof in);
    cudaMalloc((void**)&out_d, sizeof out);
    cudaMemcpy(in_d, in, sizeof in, cudaMemcpyHostToDevice);
    const char* names[] = {"printed", "threadIdx", "none"};
    int failures = 0;
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        for (int variant = 0; variant < 3; ++variant) {
            scanLanes<<<1, lanes>>>(in_d, out_d, variant);
            cudaDeviceSynchronize();
            cudaMemcpy(out, out_d, sizeof out, cudaMemcpyDeviceToHost);
            std::printf("lanes %s %s:", names[variant], order);
            for (unsigned int t = 0; t < lanes; ++t) std::printf(" %g", out[t]);
            std::printf("\n");
        }
        const unsigned int n10 = 3 * BLOCK_DIM + 37;  // a partial last block
        failures += runScan(scan_kernel_fig_11_10, "fig11.10", order, n10, (n10 + BLOCK_DIM - 1) / BLOCK_DIM, BLOCK_DIM);
        const unsigned int seg = COARSE_FACTOR * BLOCK_DIM;  // the printed kernels need whole segments
        failures += runScan(scan_kernel_fig_11_12, "fig11.12", order, 3 * seg, 3, seg);
        failures += runScan(scan_kernel_fig_11_13, "fig11.13", order, 3 * seg, 3, seg);
    }
    cudaFree(in_d);
    cudaFree(out_d);
    std::printf("%s\n", failures ? "FAILED: a printed kernel gave a wrong scan" : "DONE");
    return failures ? 1 : 0;
}
