// Runs ../wrap_pairs.cu (chapter 12's summary page) on the course's CPU emulator. For each
// of the three thread orders it prints one line per kernel:
//   "<kernel> <order>: count <idx> <val> of <want>, pairs <matched> of <count>, columns <ok>"
//   shown   the page's candidatePairs, two counters
//   fixed   the same filter taking one slot per pair from one counter, a copy made for this
//           check and not lesson code
// <idx> and <val> are the two counters after the launch (the fixed kernel has one, printed
// twice), <want> the CPU's count; a pair
// matches when outVal[k] is the logit at column outIdx[k]; <ok> is "yes" when the columns
// written are exactly the CPU's set. app/src/chapters/ch12/Wrap.snippets.test.ts reads these
// lines. On the emulator one thread runs until a barrier, a warp _sync call or its end, so
// no thread runs between another thread's two atomics.
#include <cstdio>
#include <cuda/atomic>

__global__ void candidatePairs(const float* logits, int cols, float threshold,
                               int* outIdx, float* outVal,
                               unsigned int* idxCount, unsigned int* valCount);

#if defined(PMPP_EMU)
static const int kOrders = 3;
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 11);
    return i == 0 ? "in order" : i == 1 ? "reversed" : "shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "device"; }
#endif

__global__ void candidatePairsFixed(const float* logits, int cols, float threshold,
                                    int* outIdx, float* outVal, unsigned int* count) {
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c < cols) {
        float v = logits[c];
        if (v >= threshold) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> countRef(*count);
            unsigned int k = countRef.fetch_add(1, cuda::memory_order_relaxed);   // one slot per pair
            outIdx[k] = c;
            outVal[k] = v;
        }
    }
}

int main() {
    const int cols = 1000, block = 128, grid = (cols + block - 1) / block;
    const float threshold = 6.0f;
    static float logits[1000], outVal[1000];
    static int outIdx[1000];
    static bool keptCpu[1000];
    int want = 0;
    for (int c = 0; c < cols; ++c) {
        logits[c] = (float)((c * 37 + 11) % 64) / 4.0f - 6.0f;   // -6 to 9.75, every column distinct mod 64
        keptCpu[c] = logits[c] >= threshold;
        want += keptCpu[c];
    }
    float *logits_d = nullptr, *val_d = nullptr;
    int* idx_d = nullptr;
    unsigned int* counts_d = nullptr;
    cudaMalloc((void**)&logits_d, cols * sizeof(float));
    cudaMalloc((void**)&val_d, cols * sizeof(float));
    cudaMalloc((void**)&idx_d, cols * sizeof(int));
    cudaMalloc((void**)&counts_d, 2 * sizeof(unsigned int));
    cudaMemcpy(logits_d, logits, cols * sizeof(float), cudaMemcpyHostToDevice);
    const char* names[] = {"shown", "fixed"};
    for (int k = 0; k < 2; ++k)
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            unsigned int counts[2] = {0, 0};
            for (int c = 0; c < cols; ++c) {
                outIdx[c] = -1;
                outVal[c] = -1000.0f;
            }
            cudaMemcpy(counts_d, counts, sizeof counts, cudaMemcpyHostToDevice);
            cudaMemcpy(idx_d, outIdx, cols * sizeof(int), cudaMemcpyHostToDevice);
            cudaMemcpy(val_d, outVal, cols * sizeof(float), cudaMemcpyHostToDevice);
            if (k == 0) candidatePairs<<<grid, block>>>(logits_d, cols, threshold, idx_d, val_d, counts_d, counts_d + 1);
            else candidatePairsFixed<<<grid, block>>>(logits_d, cols, threshold, idx_d, val_d, counts_d);
            cudaDeviceSynchronize();
            cudaMemcpy(counts, counts_d, sizeof counts, cudaMemcpyDeviceToHost);
            cudaMemcpy(outIdx, idx_d, cols * sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(outVal, val_d, cols * sizeof(float), cudaMemcpyDeviceToHost);
            const unsigned int n = counts[0];
            int matched = 0;
            static bool seen[1000];
            for (int c = 0; c < cols; ++c) seen[c] = false;
            bool columnsOk = (int)n == want;
            for (unsigned int s = 0; s < n && s < (unsigned)cols; ++s) {
                const int c = outIdx[s];
                if (c < 0 || c >= cols || seen[c] || !keptCpu[c]) {
                    columnsOk = false;
                    continue;
                }
                seen[c] = true;
                matched += outVal[s] == logits[c];
            }
            std::printf("%s %s: count %u %u of %d, pairs %d of %u, columns %s\n", names[k], order, counts[0],
                        k == 0 ? counts[1] : counts[0], want, matched, n, columnsOk ? "yes" : "no");
        }
    cudaFree(logits_d);
    cudaFree(val_d);
    cudaFree(idx_d);
    cudaFree(counts_d);
    std::printf("DONE\n");
    return 0;
}
