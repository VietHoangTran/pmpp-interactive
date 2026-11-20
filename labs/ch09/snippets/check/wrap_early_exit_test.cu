// Runs ../wrap_early_exit.cu (chapter 9's summary page) on the course's CPU emulator. For
// blocks of 256 and 128 threads and each of the three thread orders it prints one line per
// kernel, "<kernel> <block> <order>: wrong <k> of 256: <bin>:<got>:<want> ...", listing
// every bin whose count differs from the sequential loop's:
//   shown   the page's histogramEarlyExit
//   fixed   the same kernel with the early return replaced by a guard on the update
//           only, as in Fig. 9.10; a copy made for this check and not lesson code
// app/src/chapters/ch09/Wrap.snippets.test.ts reads these lines. The image is
// image[i] = (37 i + 11) mod 256 for 1,000 pixels, which leaves the last block short.
#include <cstdio>
#include <cuda/atomic>

#define NUM_BINS 256

__global__ void histogramEarlyExit(const unsigned char* image, unsigned int* bins, unsigned int n);

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

__global__ void histogramGuarded(const unsigned char* image, unsigned int* bins, unsigned int n) {
    __shared__ unsigned int bins_s[NUM_BINS];
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) bins_s[b] = 0;
    __syncthreads();
    unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n) {
        cuda::atomic_ref<unsigned int, cuda::thread_scope_block> r(bins_s[image[i]]);
        r.fetch_add(1, cuda::memory_order_relaxed);
    }
    __syncthreads();
    for(unsigned int b = threadIdx.x; b < NUM_BINS; b += blockDim.x) {
        if(bins_s[b] > 0) {
            cuda::atomic_ref<unsigned int, cuda::thread_scope_device> r(bins[b]);
            r.fetch_add(bins_s[b], cuda::memory_order_relaxed);
        }
    }
}

int main() {
    const unsigned int n = 1000;
    static unsigned char image[1000];
    static unsigned int want[NUM_BINS], got[NUM_BINS];
    for (unsigned int i = 0; i < n; ++i) {
        image[i] = (unsigned char)((37u * i + 11u) % 256u);
        ++want[image[i]];
    }
    unsigned char* image_d = nullptr;
    unsigned int* bins_d = nullptr;
    cudaMalloc((void**)&image_d, n);
    cudaMalloc((void**)&bins_d, NUM_BINS * sizeof(unsigned int));
    cudaMemcpy(image_d, image, n, cudaMemcpyHostToDevice);
    const unsigned int blocks[] = {256, 128};
    const char* names[] = {"shown", "fixed"};
    for (int b = 0; b < 2; ++b)
        for (int k = 0; k < 2; ++k)
            for (int o = 0; o < kOrders; ++o) {
                const char* order = useOrder(o);
                cudaMemset(bins_d, 0, NUM_BINS * sizeof(unsigned int));
                unsigned int grid = (n + blocks[b] - 1) / blocks[b];
                if (k == 0) histogramEarlyExit<<<grid, blocks[b]>>>(image_d, bins_d, n);
                else histogramGuarded<<<grid, blocks[b]>>>(image_d, bins_d, n);
                cudaDeviceSynchronize();
                cudaMemcpy(got, bins_d, NUM_BINS * sizeof(unsigned int), cudaMemcpyDeviceToHost);
                int wrong = 0;
                for (int v = 0; v < NUM_BINS; ++v) wrong += got[v] != want[v];
                std::printf("%s %u %s: wrong %d of %d:", names[k], blocks[b], order, wrong, NUM_BINS);
                for (int v = 0; v < NUM_BINS; ++v)
                    if (got[v] != want[v]) std::printf(" %d:%u:%u", v, got[v], want[v]);
                std::printf("\n");
            }
    cudaFree(image_d);
    cudaFree(bins_d);
    std::printf("DONE\n");
    return 0;
}
