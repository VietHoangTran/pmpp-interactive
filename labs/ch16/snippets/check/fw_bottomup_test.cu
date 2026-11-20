// Runs ../fw_bottomup.cu, the corrected Fig. 16.4 shown in section 16.3, against a
// sequential CPU Floyd-Warshall:
//  - on Fig. 16.5's table as printed (b to f = 9), one step at a time, checking every
//    intermediate table and the step-c changes the lesson states (a to e 23, b to e 19,
//    g to e 23, h to e 13, after steps a and b);
//  - on random graphs with non-negative weights, through the host loop floydWarshall(),
//    with block sizes that leave threads out of range in the last block of every row;
// each under the emulator's three thread orders.
#include <climits>
#include <cstdint>
#include <cstdio>
#include <vector>
#include "check.h"

#define FW_INF (INT_MAX / 2)  // the sentinel of ../fw_bottomup.cu; a mismatch fails every case with a missing edge

__global__ void FW_bottomup(int k, int* dist, int n_vertices);
void floydWarshall(int* dist_d, int n_vertices, int threads);

using Table = std::vector<int>;

static void cpuStep(Table& d, int n, int k) {
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            int ik = d[i * n + k], kj = d[k * n + j];
            if (ik != FW_INF && kj != FW_INF && ik + kj < d[i * n + j]) d[i * n + j] = ik + kj;
        }
}

static Table fig16_5() {
    const int n = 8;  // a b c d e f g h
    Table d(n * n, FW_INF);
    for (int i = 0; i < n; ++i) d[i * n + i] = 0;
    const int e[][3] = {{0, 1, 4}, {0, 6, 7}, {0, 7, 4}, {1, 2, 9}, {1, 5, 9}, {1, 6, 8}, {1, 7, 1}, {2, 4, 10},
                        {4, 2, 8}, {4, 3, 6}, {4, 5, 5}, {5, 4, 6}, {6, 1, 4}, {6, 5, 7}, {7, 2, 3}};
    for (const auto& x : e) d[x[0] * n + x[1]] = x[2];
    return d;
}

static Table randomGraph(uint32_t& x, int n, int densityPercent) {
    Table d(n * n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j) {
            x = x * 1664525u + 1013904223u;
            bool edge = (int)((x >> 16) % 100u) < densityPercent;
            x = x * 1664525u + 1013904223u;
            d[i * n + j] = i == j ? 0 : edge ? (int)((x >> 16) % 21u) : FW_INF;
        }
    return d;
}

static void stepByStep(int threads) {
    const int n = 8;
    const Table start = fig16_5();
    int* d_d = nullptr;
    cudaMalloc((void**)&d_d, n * n * sizeof(int));
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        Table ref = start, got(n * n);
        cudaMemcpy(d_d, start.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
        bool same = true;
        int changedAtC = -1;
        bool cellsAtC = false;
        for (int k = 0; k < n; ++k) {
            Table before = ref;
            cpuStep(ref, n, k);
            dim3 block(threads), grid((n + threads - 1) / threads, n);
            FW_bottomup<<<grid, block>>>(k, d_d, n);
            cudaDeviceSynchronize();  // the check reads the table back after every step
            cudaMemcpy(got.data(), d_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);
            same = same && got == ref;
            if (k == 2) {
                changedAtC = 0;
                for (int i = 0; i < n * n; ++i) changedAtC += got[i] != before[i];
                cellsAtC = got[0 * n + 4] == 23 && got[1 * n + 4] == 19 && got[6 * n + 4] == 23 && got[7 * n + 4] == 13;
            }
        }
        char what[200];
        std::snprintf(what, sizeof what, "Fig. 16.5, %d threads per block, %s: every step equals the CPU step", threads, order);
        expect(same, what);
        std::snprintf(what, sizeof what, "Fig. 16.5, %d threads per block, %s: step c changes 4 cells, a-e 23, b-e 19, g-e 23, h-e 13", threads, order);
        expect(changedAtC == 4 && cellsAtC, what);
    }
    cudaFree(d_d);
}

static void wholeRun(uint32_t& x, int n, int threads, int density) {
    const Table start = randomGraph(x, n, density);
    Table ref = start;
    for (int k = 0; k < n; ++k) cpuStep(ref, n, k);
    int* d_d = nullptr;
    cudaMalloc((void**)&d_d, n * n * sizeof(int));
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        Table got(n * n);
        cudaMemcpy(d_d, start.data(), n * n * sizeof(int), cudaMemcpyHostToDevice);
        floydWarshall(d_d, n, threads);
        cudaMemcpy(got.data(), d_d, n * n * sizeof(int), cudaMemcpyDeviceToHost);
        char what[200];
        std::snprintf(what, sizeof what, "random graph, %d vertices, %d%% edges, %d threads per block, %s: equals the CPU", n, density, threads, order);
        expect(got == ref, what);
    }
    cudaFree(d_d);
}

int main() {
    std::printf("fw_bottomup.cu\n");
    for (int threads : {8, 3, 32}) stepByStep(threads);
    uint32_t x = 1616u;
    const int sizes[][3] = {{1, 1, 50}, {5, 4, 40}, {17, 4, 30}, {17, 32, 30}, {40, 16, 10}, {40, 64, 60}, {64, 32, 5}};
    for (const auto& s : sizes) wholeRun(x, s[0], s[1], s[2]);
    return finish();
}
