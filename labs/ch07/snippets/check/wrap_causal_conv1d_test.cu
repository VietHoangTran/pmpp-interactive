// Runs ../wrap_causal_conv1d.cu (the kernel of the summary page's first question) on the
// course's CPU emulator for every length n from 1 to 400, in the emulator's three thread
// orders, against a CPU reference. Beside it runs a fixed copy made for this check, not
// lesson code: the fix the page describes (no early return; every thread loads, a ghost
// zero past the end, and reaches the barrier; only then is i < n tested). It prints one
// line per kernel and order with the lengths for which any output was wrong:
//   "<shown|fixed> <order>: wrong for n =<list>"
// which app/src/chapters/ch07/Wrap.test.ts reads. Inputs and weights are small whole
// numbers and quarters, so every sum is exact and the comparison can be exact. y starts
// as NaN, so an output nobody writes is counted wrong.
#include <cstdio>
#include <limits>

#include "../wrap_causal_conv1d.cu"

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

__global__ void causalConv1dFixed(const float* x, float* y, int n) {
    __shared__ float tile[BLOCK + K - 1];
    int t = threadIdx.x;
    int i = blockIdx.x * BLOCK + t;
    tile[t + K - 1] = (i < n) ? x[i] : 0.0f;   // past the end: a ghost zero
    if (t < K - 1) {
        int j = i - (K - 1);
        tile[t] = (j >= 0) ? x[j] : 0.0f;
    }
    __syncthreads();                           // every thread gets here
    if (i >= n) return;
    float acc = 0.0f;
    for (int k = 0; k < K; k++) acc += w[k] * tile[t + k];
    y[i] = acc;
}

int main() {
    const int maxN = 400;
    const float wh[K] = {0.5f, -1.0f, 2.0f, 0.25f};
    cudaMemcpyToSymbol(w, wh, sizeof(wh));
    static float x[maxN], y[maxN], want[maxN];
    float *x_d = nullptr, *y_d = nullptr;
    cudaMalloc((void**)&x_d, maxN * sizeof(float));
    cudaMalloc((void**)&y_d, maxN * sizeof(float));
    const char* names[] = {"shown", "fixed"};
    for (int k = 0; k < 2; ++k)
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            std::printf("%s %s: wrong for n =", names[k], order);
            for (int n = 1; n <= maxN; ++n) {
                for (int i = 0; i < n; ++i) x[i] = (float)(((i * 37) % 11) - 5);
                for (int i = 0; i < n; ++i) {
                    float acc = 0.0f;
                    for (int q = 0; q < K; ++q) {
                        int j = i - (K - 1) + q;
                        if (j >= 0) acc += wh[q] * x[j];
                    }
                    want[i] = acc;
                    y[i] = std::numeric_limits<float>::quiet_NaN();
                }
                cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
                cudaMemcpy(y_d, y, n * sizeof(float), cudaMemcpyHostToDevice);
                int blocks = (n + BLOCK - 1) / BLOCK;
                if (k == 0) causalConv1d<<<blocks, BLOCK>>>(x_d, y_d, n);
                else causalConv1dFixed<<<blocks, BLOCK>>>(x_d, y_d, n);
                cudaDeviceSynchronize();
                cudaMemcpy(y, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
                bool ok = true;
                for (int i = 0; i < n; ++i) ok = ok && (y[i] == want[i]);
                if (!ok) std::printf(" %d", n);
            }
            std::printf("\n");
        }
    cudaFree(x_d);
    cudaFree(y_d);
    return 0;
}
