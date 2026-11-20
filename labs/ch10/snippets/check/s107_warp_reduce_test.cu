// Runs ../s107_warp_reduce_as_printed.cu (Fig. 10.13 as printed) on the course's CPU
// emulator for section 10.7, and prints what the lanes return so that
// app/src/chapters/ch10/S107.snippets.test.ts can compare it with the model behind the
// section's interactives (app/src/sim/shuffle.ts). One warp of 32 lanes, lane i passes i.
//   full: every lane's return value, under each of the three thread orders
//   return N: lane 0's result when lanes N to 31 return before calling warp_reduce
//   zero N: lane 0's result when lanes N to 31 stay and pass 0 instead
// The snippet has no kernel and no shared memory, so it is included as it is.
#include <cstdio>
#include "../s107_warp_reduce_as_printed.cu"

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

// mode 0: every lane calls. 1: lanes from n on return first. 2: they pass 0 instead.
__global__ void reduceLanes(const float* in, float* out, int n, int mode) {
    unsigned int t = threadIdx.x;
    if (mode == 1 && t >= (unsigned int)n) return;
    float v = (mode == 2 && t >= (unsigned int)n) ? 0.0f : in[t];
    out[t] = warp_reduce(v);
}

static void run(const float* in_d, float* out_d, float* out, int n, int mode) {
    for (int i = 0; i < WARP_SIZE; ++i) out[i] = -1.0f;
    cudaMemcpy(out_d, out, WARP_SIZE * sizeof(float), cudaMemcpyHostToDevice);
    reduceLanes<<<1, WARP_SIZE>>>(in_d, out_d, n, mode);
    cudaDeviceSynchronize();
    cudaMemcpy(out, out_d, WARP_SIZE * sizeof(float), cudaMemcpyDeviceToHost);
}

int main() {
    float in[WARP_SIZE], out[WARP_SIZE], first[WARP_SIZE];
    for (int i = 0; i < WARP_SIZE; ++i) in[i] = (float)i;
    float *in_d = nullptr, *out_d = nullptr;
    cudaMalloc((void**)&in_d, sizeof in);
    cudaMalloc((void**)&out_d, sizeof out);
    cudaMemcpy(in_d, in, sizeof in, cudaMemcpyHostToDevice);
    int failures = 0;
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        run(in_d, out_d, out, WARP_SIZE, 0);
        std::printf("full %s:", order);
        for (int i = 0; i < WARP_SIZE; ++i) std::printf(" %g", out[i]);
        std::printf("\n");
        if (o == 0)
            for (int i = 0; i < WARP_SIZE; ++i) first[i] = out[i];
        else
            for (int i = 0; i < WARP_SIZE; ++i) failures += out[i] != first[i];
    }
    const int ns[] = {20, 1};
    for (int n : ns) {
        run(in_d, out_d, out, n, 1);
        std::printf("return %d: %g\n", n, out[0]);
        run(in_d, out_d, out, n, 2);
        std::printf("zero %d: %g\n", n, out[0]);
    }
    cudaFree(in_d);
    cudaFree(out_d);
    std::printf("%s\n", failures ? "FAILED: thread orders disagree" : "DONE");
    return failures ? 1 : 0;
}
