// Runs ../draw_search.cu (section X1.8) on the course's CPU emulator and prints what
// app/src/sim/draw.test.ts compares with app/src/sim/draw.ts. The rows come from the file
// named on the command line, written by the test:
//   first line:  <rows> <stride>
//   then a line per row:  <n> <u as float bits> <n weights as float bits>   (hexadecimal)
// Output, under each of the emulator's thread orders (one order on the race flavour):
//   order <name>
//   token <row> <position>
//   error <message>          if a launch failed
#include <cstdio>
#include <cstring>
#include <vector>
#include "../draw_search.cu"

#if defined(PMPP_EMU) && !defined(PMPP_EMU_RACE)
static const int kOrders = 3;
// The same three orders, and the same shuffle seed, as labs/common/src/backend_cudalike.cpp.
static const char* useOrder(int i) {
    using pmpp_emu::Schedule;
    pmpp_emu::set_schedule(i == 0 ? Schedule::Forward : i == 1 ? Schedule::Reverse : Schedule::Shuffle, 7);
    return i == 0 ? "in order" : i == 1 ? "reversed" : "shuffled";
}
#else
static const int kOrders = 1;
static const char* useOrder(int) { return "concurrent"; }
#endif

static float asFloat(unsigned int b) {
    float f;
    std::memcpy(&f, &b, sizeof f);
    return f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s rows.txt\n", argv[0]);
        return 2;
    }
    FILE* f = std::fopen(argv[1], "r");
    if (!f) return 2;
    int rows = 0, stride = 0;
    if (std::fscanf(f, "%d %d", &rows, &stride) != 2) return 2;
    std::vector<float> w((size_t)rows * stride, 0.0f), u(rows);
    std::vector<int> counts(rows);
    for (int r = 0; r < rows; ++r) {
        unsigned int b;
        if (std::fscanf(f, "%d %x", &counts[r], &b) != 2) return 2;
        u[r] = asFloat(b);
        for (int i = 0; i < counts[r]; ++i) {
            if (std::fscanf(f, "%x", &b) != 1) return 2;
            w[(size_t)r * stride + i] = asFloat(b);
        }
    }
    std::fclose(f);

    float *w_d, *u_d;
    int *counts_d, *tokens_d;
    cudaMalloc((void**)&w_d, w.size() * sizeof(float));
    cudaMalloc((void**)&u_d, rows * sizeof(float));
    cudaMalloc((void**)&counts_d, rows * sizeof(int));
    cudaMalloc((void**)&tokens_d, rows * sizeof(int));
    cudaMemcpy(w_d, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(u_d, u.data(), rows * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(counts_d, counts.data(), rows * sizeof(int), cudaMemcpyHostToDevice);
    std::vector<int> tokens(rows);
    for (int o = 0; o < kOrders; ++o) {
        std::printf("order %s\n", useOrder(o));
        cudaMemset(tokens_d, 0xff, rows * sizeof(int));
        launchDraw(w_d, counts_d, u_d, tokens_d, rows, stride);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) std::printf("error %s\n", cudaGetErrorString(e));
        cudaMemcpy(tokens.data(), tokens_d, rows * sizeof(int), cudaMemcpyDeviceToHost);
        for (int r = 0; r < rows; ++r) std::printf("token %d %d\n", r, tokens[r]);
    }
    cudaFree(w_d);
    cudaFree(u_d);
    cudaFree(counts_d);
    cudaFree(tokens_d);
    std::printf("DONE\n");
    return 0;
}
