// Runs ../merge_basic.cu, the corrected Figs. 13.2, 13.5 and 13.9 shown in section 13.3,
// against std::merge on sorted int arrays with many duplicate keys, under several launch
// shapes, including the question's 1,000 outputs on 3 blocks of 128 threads. It also
// checks that nothing at or past C[m + n] is written. Plain ints: this cannot see
// stability (lab 13.1 merges key-value pairs for that).
#include <algorithm>
#include <cstdint>
#include <vector>
#include "check.h"

__global__ void merge_basic_kernel(const int* A, int m, const int* B, int n, int* C);

static std::vector<int> sortedRandom(uint32_t& x, int len, int keys) {
    std::vector<int> v(len);
    for (int& e : v) { x = x * 1664525u + 1013904223u; e = (int)((x >> 16) % (uint32_t)keys); }
    std::sort(v.begin(), v.end());
    return v;
}

static void runCase(const char* name, const std::vector<int>& a, const std::vector<int>& b, int blocks, int threads) {
    const int m = (int)a.size(), n = (int)b.size(), pad = 8;
    std::vector<int> ref(m + n);
    std::merge(a.begin(), a.end(), b.begin(), b.end(), ref.begin());
    int *A_d = nullptr, *B_d = nullptr, *C_d = nullptr;
    cudaMalloc((void**)&A_d, (m + 1) * sizeof(int));
    cudaMalloc((void**)&B_d, (n + 1) * sizeof(int));
    cudaMalloc((void**)&C_d, (m + n + pad) * sizeof(int));
    if (m) cudaMemcpy(A_d, a.data(), m * sizeof(int), cudaMemcpyHostToDevice);
    if (n) cudaMemcpy(B_d, b.data(), n * sizeof(int), cudaMemcpyHostToDevice);
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<int> got(m + n + pad, -7);
        cudaMemcpy(C_d, got.data(), got.size() * sizeof(int), cudaMemcpyHostToDevice);
        merge_basic_kernel<<<blocks, threads>>>(A_d, m, B_d, n, C_d);
        cudaDeviceSynchronize();
        cudaMemcpy(got.data(), C_d, got.size() * sizeof(int), cudaMemcpyDeviceToHost);
        bool same = std::equal(ref.begin(), ref.end(), got.begin());
        bool tail = std::all_of(got.begin() + m + n, got.end(), [](int v) { return v == -7; });
        char what[200];
        std::snprintf(what, sizeof what, "%s (m %d, n %d, %d x %d threads), %s: equals std::merge, nothing written past the end",
                      name, m, n, blocks, threads, order);
        expect(same && tail, what);
    }
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
}

int main() {
    std::printf("merge_basic.cu\n");
    runCase("running example", {1, 7, 8, 9, 10}, {7, 10, 10, 12}, 1, 3);
    runCase("running example", {1, 7, 8, 9, 10}, {7, 10, 10, 12}, 1, 4);
    runCase("running example, more threads than outputs", {1, 7, 8, 9, 10}, {7, 10, 10, 12}, 2, 8);
    uint32_t x = 2027u;
    {
        std::vector<int> a = sortedRandom(x, 500, 40), b = sortedRandom(x, 500, 40);
        runCase("1,000 outputs on 3 blocks of 128", a, b, 3, 128);
    }
    const int shapes[][2] = {{1, 1}, {1, 32}, {2, 32}, {4, 64}};
    const int sizes[][2] = {{0, 0}, {0, 37}, {41, 0}, {1, 1}, {63, 65}, {200, 3}};
    for (const auto& s : sizes)
        for (const auto& g : shapes) {
            std::vector<int> a = sortedRandom(x, s[0], 5), b = sortedRandom(x, s[1], 5);
            runCase("random, keys 0 to 4", a, b, g[0], g[1]);
        }
    return finish();
}
