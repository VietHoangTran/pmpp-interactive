// Runs ../wrap_centre.cu (chapter 10's summary page) on the course's CPU emulator. For
// blocks of 256 and 64 threads and each of the three thread orders it prints one line per
// kernel, "<kernel> <block> <order>: right <k> of <n>: <row>:<col> ...", listing every
// output that equals the CPU's x - mean(x) exactly:
//   shown   the page's centreRows
//   fixed   the same kernel with the row mean handed back through shared memory after a
//           barrier, a copy made for this check and not lesson code
// app/src/chapters/ch10/Wrap.snippets.test.ts reads these lines. The data are small whole
// numbers, so every sum is exact and the comparison can be exact.
#include <cstdio>

__global__ void centreRows(const float* x, float* y, int cols);

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

// The check's own copy of the warp sum (the emulator models 32-lane warps).
__device__ __inline__ float warpSum(float v) {
    for (unsigned int s = warpSize / 2; s > 0; s /= 2) v += __shfl_down_sync(~0u, v, s);
    return v;
}

__global__ void centreRowsFixed(const float* x, float* y, int cols) {
    const float* row = x + blockIdx.x * cols;
    float* out = y + blockIdx.x * cols;
    unsigned int lane = threadIdx.x % warpSize, warp = threadIdx.x / warpSize;
    float s = 0.0f;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) s += row[c];
    s = warpSum(s);
    __shared__ float partial_s[32];
    if (lane == 0) partial_s[warp] = s;
    __syncthreads();
    __shared__ float mean_s;
    if (warp == 0) {
        unsigned int warps = blockDim.x / warpSize;
        s = lane < warps ? partial_s[lane] : 0.0f;
        s = warpSum(s);
        if (threadIdx.x == 0) mean_s = s / cols;   // hand the block's result back
    }
    __syncthreads();
    float mean = mean_s;
    for (int c = threadIdx.x; c < cols; c += blockDim.x) out[c] = row[c] - mean;
}

int main() {
    const int rows = 3, cols = 1000, n = rows * cols;
    static float x[3 * 1000], y[3 * 1000], want[3 * 1000];
    for (int r = 0; r < rows; ++r) {
        float s = 0.0f;
        for (int c = 0; c < cols; ++c) {
            x[r * cols + c] = (float)(((r * 7 + c * 3) % 11) + 1);   // 1 to 11: exact sums
            s += x[r * cols + c];
        }
        float mean = s / cols;
        for (int c = 0; c < cols; ++c) want[r * cols + c] = x[r * cols + c] - mean;
    }
    float *x_d = nullptr, *y_d = nullptr;
    cudaMalloc((void**)&x_d, n * sizeof(float));
    cudaMalloc((void**)&y_d, n * sizeof(float));
    cudaMemcpy(x_d, x, n * sizeof(float), cudaMemcpyHostToDevice);
    const int blocks[] = {256, 64};
    const char* names[] = {"shown", "fixed"};
    for (int b = 0; b < 2; ++b)
        for (int k = 0; k < 2; ++k)
            for (int o = 0; o < kOrders; ++o) {
                const char* order = useOrder(o);
                for (int i = 0; i < n; ++i) y[i] = -1.0f;
                cudaMemcpy(y_d, y, n * sizeof(float), cudaMemcpyHostToDevice);
                if (k == 0) centreRows<<<rows, blocks[b]>>>(x_d, y_d, cols);
                else centreRowsFixed<<<rows, blocks[b]>>>(x_d, y_d, cols);
                cudaDeviceSynchronize();
                cudaMemcpy(y, y_d, n * sizeof(float), cudaMemcpyDeviceToHost);
                int right = 0;
                for (int i = 0; i < n; ++i) right += y[i] == want[i];
                std::printf("%s %d %s: right %d of %d:", names[k], blocks[b], order, right, n);
                if (k == 0)
                    for (int i = 0; i < n; ++i)
                        if (y[i] == want[i]) std::printf(" %d:%d", i / cols, i % cols);
                std::printf("\n");
            }
    cudaFree(x_d);
    cudaFree(y_d);
    std::printf("DONE\n");
    return 0;
}
