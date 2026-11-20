// Runs ../s75_halo_toy.cu (section 7.5's stepper: the corrected Fig. 7.12 kernel at the
// sizes of Fig. 7.13, radius 1, an 8 x 8 block and a 6 x 6 output tile) on the course's
// CPU emulator, through the snippet's own launchHaloToy, on several image shapes and in
// the emulator's three thread orders, against a CPU reference with zero ghost cells. It
// prints one line per case,
//   "halo <W>x<H> <order>: right <k> of <n>"
// which app/src/chapters/ch07/S75.snippets.test.ts reads. Inputs and weights are small
// whole numbers, so every sum is exact and the comparison can be exact. P starts as NaN,
// so an output nobody writes is counted wrong.
#include <cstdio>
#include <limits>

#include "../s75_halo_toy.cu"

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

int main() {
    const int D = 2 * FILTER_RADIUS + 1;
    // Lopsided on purpose, as in the stepper, so that a shifted patch gives a different sum.
    const float F[2 * FILTER_RADIUS + 1][2 * FILTER_RADIUS + 1] = {{1, 2, 0}, {0, 3, 1}, {2, 0, 1}};
    cudaMemcpyToSymbol(F_c, F, sizeof(F));
    const int shapes[][2] = {{10, 10}, {14, 14}, {13, 7}, {6, 6}, {1, 1}, {37, 19}};
    for (const auto& s : shapes) {
        const int W = s[0], H = s[1], n = W * H;
        float* x = new float[n];
        float* y = new float[n];
        float* want = new float[n];
        for (int i = 0; i < n; ++i) x[i] = (float)(((i * 7) % 11) - 5);
        for (int r = 0; r < H; ++r)
            for (int c = 0; c < W; ++c) {
                float acc = 0.0f;
                for (int fr = 0; fr < D; ++fr)
                    for (int fc = 0; fc < D; ++fc) {
                        int rr = r - FILTER_RADIUS + fr, cc = c - FILTER_RADIUS + fc;
                        if (rr >= 0 && rr < H && cc >= 0 && cc < W) acc += F[fr][fc] * x[rr * W + cc];
                    }
                want[r * W + c] = acc;
            }
        float *x_d = nullptr, *y_d = nullptr;
        cudaMalloc((void**)&x_d, n * sizeof(float));
        cudaMalloc((void**)&y_d, n * sizeof(float));
        cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            for (int i = 0; i < n; ++i) y[i] = std::numeric_limits<float>::quiet_NaN();
            cudaMemcpy(y_d, y, n * sizeof(float), cudaMemcpyHostToDevice);
            launchHaloToy(x_d, y_d, W, H);
            cudaDeviceSynchronize();
            cudaMemcpy(y, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
            int right = 0;
            for (int i = 0; i < n; ++i) right += (y[i] == want[i]);
            std::printf("halo %dx%d %s: right %d of %d\n", W, H, order, right, n);
        }
        cudaFree(x_d);
        cudaFree(y_d);
        delete[] x;
        delete[] y;
        delete[] want;
    }
    return 0;
}
