// Runs ../wrap_mass_before.cu (chapter 11's summary page) on the course's CPU emulator. For
// three shapes (blocks of 256 threads on rows of 256 and of 200 tokens, blocks of 64 on rows
// of 64) and each of the three thread orders it prints one line per kernel,
// "<kernel> <block> <cols> <order>: wrong <k> of <n>: <row>:<col> ...", listing every output
// that differs from the CPU's exclusive scan:
//   shown   the page's massBeforeRows
//   fixed   the same kernel with lane 0 of every warp taking the scanned total of the warps
//           before it, a copy made for this check and not lesson code
// Each line ends with "| inclusive <m>": how many of the wrong outputs hold the inclusive
// scan instead, the token's own probability included.
// app/src/chapters/ch11/Wrap.snippets.test.ts reads these lines. The data are small whole
// numbers, so every sum is exact and the comparison can be exact.
#include <cstdio>

__global__ void massBeforeRows(const float* probs, float* massBefore, unsigned int cols);

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

// The check's own copies of the warp and block scans (the emulator models 32-lane warps).
__device__ float checkWarpScan(float v) {
    unsigned int lane = threadIdx.x % warpSize;
    for (unsigned int s = 1; s < (unsigned int)warpSize; s *= 2) {
        float left = __shfl_up_sync(~0u, v, s);
        if (lane >= s) v += left;
    }
    return v;
}

__global__ void massBeforeRowsFixed(const float* probs, float* massBefore, unsigned int cols) {
    __shared__ float warpSums_s[32];
    const float* row = probs + blockIdx.x * cols;
    float* out = massBefore + blockIdx.x * cols;
    unsigned int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
    float p = (threadIdx.x < cols) ? row[threadIdx.x] : 0.0f;
    float v = checkWarpScan(p);
    if (lane == (unsigned int)warpSize - 1) warpSums_s[warp] = v;
    __syncthreads();
    if (warp == 0) {
        unsigned int warps = blockDim.x / warpSize;
        float s = lane < warps ? warpSums_s[lane] : 0.0f;
        s = checkWarpScan(s);
        if (lane < warps) warpSums_s[lane] = s;
    }
    __syncthreads();
    if (warp > 0) v += warpSums_s[warp - 1];
    float before = __shfl_up_sync(~0u, v, 1);
    if (lane == 0) before = warp > 0 ? warpSums_s[warp - 1] : 0.0f;   // the warps before this one
    if (threadIdx.x < cols) out[threadIdx.x] = before;
}

int main() {
    const int rows = 3;
    struct Shape { int block, cols; };
    const Shape shapes[] = {{256, 256}, {256, 200}, {64, 64}};
    static float x[3 * 256], y[3 * 256], want[3 * 256];
    float *x_d = nullptr, *y_d = nullptr;
    cudaMalloc((void**)&x_d, sizeof(x));
    cudaMalloc((void**)&y_d, sizeof(y));
    const char* names[] = {"shown", "fixed"};
    for (const Shape& sh : shapes) {
        const int n = rows * sh.cols;
        for (int r = 0; r < rows; ++r) {
            float s = 0.0f;
            for (int c = 0; c < sh.cols; ++c) {
                x[r * sh.cols + c] = (float)(((r * 5 + c * 7) % 9) + 1);   // 1 to 9: exact sums
                want[r * sh.cols + c] = s;                                  // the exclusive scan
                s += x[r * sh.cols + c];
            }
        }
        cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
        for (int k = 0; k < 2; ++k)
            for (int o = 0; o < kOrders; ++o) {
                const char* order = useOrder(o);
                for (int i = 0; i < n; ++i) y[i] = -1.0f;
                cudaMemcpy(y_d, y, n * sizeof(float), cudaMemcpyHostToDevice);
                if (k == 0) massBeforeRows<<<rows, sh.block>>>(x_d, y_d, (unsigned int)sh.cols);
                else massBeforeRowsFixed<<<rows, sh.block>>>(x_d, y_d, (unsigned int)sh.cols);
                cudaDeviceSynchronize();
                cudaMemcpy(y, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
                int wrong = 0;
                for (int i = 0; i < n; ++i) wrong += y[i] != want[i];
                std::printf("%s %d %d %s: wrong %d of %d:", names[k], sh.block, sh.cols, order, wrong, n);
                int inclusive = 0;
                for (int i = 0; i < n; ++i)
                    if (y[i] != want[i]) {
                        std::printf(" %d:%d", i / sh.cols, i % sh.cols);
                        inclusive += y[i] == want[i] + x[i];
                    }
                std::printf(" | inclusive %d\n", inclusive);
            }
    }
    cudaFree(x_d);
    cudaFree(y_d);
    std::printf("DONE\n");
    return 0;
}
