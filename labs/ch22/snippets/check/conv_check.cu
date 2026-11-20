// Runs ../scatter_conv_plain.cu and ../gather_conv.cu (section 22.2, "This scatter passes on
// the emulator") on the course's CPU emulators. app/src/chapters/ch22/S222.snippets.test.ts
// appends this file to the two snippets, translates the result with labs/tools/cu2emu.py and
// builds it twice, as the lab runner builds the `emu` and `race` backends.
//
// Usage: conv_check [gather|plain|atomic|all]. For each kernel and each thread order it
// prints "<kernel> <order>: wrong <k> of <n>", comparing with a sequential loop, then "DONE".
//   gather  the lesson's convGather
//   plain   the lesson's convScatterPlain (the += is not atomic)
//   atomic  the same scatter with atomicAdd; a copy made for this check, not lesson code
// On `emu` there are three orders (in order, reversed, shuffled); on `race` one concurrent
// run, and ThreadSanitizer reports any unsynchronised pair of accesses on stderr.
// Inputs and taps are small integers, so every sum is exact in any order.
#include <cstdio>
#include <cstring>

__global__ void convScatterPlain(const float* in, const float* filter, float* out, int n, int r);
__global__ void convGather(const float* in, const float* filter, float* out, int n, int r);

__global__ void convScatterAtomic(const float* in, const float* filter, float* out, int n, int r) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = in[i];
    for (int t = 0; t <= 2 * r; ++t) {
        int o = i - t + r;
        if (o >= 0 && o < n) atomicAdd(&out[o], filter[t] * v);
    }
}

#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
static const int kOrders = 3;
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "forward" : i == 1 ? "reverse" : "shuffle";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "concurrent"; }
#endif

int main(int argc, char** argv) {
    const char* only = argc > 1 ? argv[1] : "all";
    const int n = 48, r = 2, block = 16;
    static float in[48], want[48], got[48];
    const float filter[5] = {1, 2, 3, 2, 1};
    for (int i = 0; i < n; ++i) in[i] = (float)((i * 7 + 3) % 11 - 5);
    for (int o = 0; o < n; ++o) {
        float s = 0;
        for (int t = 0; t <= 2 * r; ++t) {
            int i = o + t - r;
            if (i >= 0 && i < n) s += filter[t] * in[i];
        }
        want[o] = s;
    }
    float *in_d = nullptr, *filter_d = nullptr, *out_d = nullptr;
    cudaMalloc((void**)&in_d, n * sizeof(float));
    cudaMalloc((void**)&filter_d, 5 * sizeof(float));
    cudaMalloc((void**)&out_d, n * sizeof(float));
    cudaMemcpy(in_d, in, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(filter_d, filter, 5 * sizeof(float), cudaMemcpyHostToDevice);
    const char* names[] = {"gather", "plain", "atomic"};
    for (int k = 0; k < 3; ++k) {
        if (std::strcmp(only, "all") != 0 && std::strcmp(only, names[k]) != 0) continue;
        for (int ord = 0; ord < kOrders; ++ord) {
            const char* order = useOrder(ord);
            cudaMemset(out_d, 0, n * sizeof(float));
            if (k == 0) convGather<<<n / block, block>>>(in_d, filter_d, out_d, n, r);
            if (k == 1) convScatterPlain<<<n / block, block>>>(in_d, filter_d, out_d, n, r);
            if (k == 2) convScatterAtomic<<<n / block, block>>>(in_d, filter_d, out_d, n, r);
            cudaDeviceSynchronize();
            cudaMemcpy(got, out_d, n * sizeof(float), cudaMemcpyDeviceToHost);
            int wrong = 0;
            for (int o = 0; o < n; ++o) wrong += got[o] != want[o];
            std::printf("%s %s: wrong %d of %d\n", names[k], order, wrong, n);
        }
    }
    cudaFree(in_d);
    cudaFree(filter_d);
    cudaFree(out_d);
    std::printf("DONE\n");
    return 0;
}
