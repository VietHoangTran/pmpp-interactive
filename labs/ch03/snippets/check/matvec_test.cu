// Runs two of the four kernels of ../matvec_options.cu (book exercise 2): option 2, the
// answer, against a = B c computed on the host, and option 3 against Bt c. N is not a
// multiple of the block, so the grid has surplus threads.
#include <cmath>
#include <cstdint>
#include <vector>
#include "check.h"

__global__ void matvecOption2(float* a, const float* B, const float* c, int N);
__global__ void matvecOption3(float* a, const float* B, const float* c, int N);

// Largest difference, relative to the size of the terms that were summed.
static double worst(const std::vector<float>& got, const std::vector<double>& want, const std::vector<double>& mag) {
    double w = 0;
    for (size_t i = 0; i < got.size(); ++i) w = std::fmax(w, std::fabs(got[i] - want[i]) / (mag[i] + 1e-30));
    return w;
}

int main() {
    const int N = 37;
    std::vector<float> B(N * N), c(N);
    uint32_t x = 4242u;
    auto next = [&] { x = x * 1664525u + 1013904223u; return (float)((x >> 8) & 0xffff) / 65536.0f - 0.5f; };
    for (float& v : B) v = next();
    for (float& v : c) v = next();
    std::vector<double> Bc(N, 0), Btc(N, 0), magBc(N, 0), magBtc(N, 0);
    for (int i = 0; i < N; ++i)
        for (int j = 0; j < N; ++j) {
            Bc[i] += (double)B[i * N + j] * c[j];
            magBc[i] += std::fabs((double)B[i * N + j] * c[j]);
            Btc[i] += (double)B[j * N + i] * c[j];
            magBtc[i] += std::fabs((double)B[j * N + i] * c[j]);
        }
    float *B_d = nullptr, *c_d = nullptr, *a_d = nullptr;
    cudaMalloc((void**)&B_d, N * N * sizeof(float));
    cudaMalloc((void**)&c_d, N * sizeof(float));
    cudaMalloc((void**)&a_d, N * sizeof(float));
    cudaMemcpy(B_d, B.data(), N * N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(c_d, c.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    std::printf("matvec_options.cu, N = %d, blocks of 16\n", N);
    for (int o = 0; o < kOrders; ++o) {
        const char* order = useOrder(o);
        std::vector<float> a(N);
        char what[160];
        matvecOption2<<<(N + 15) / 16, 16>>>(a_d, B_d, c_d, N);
        cudaDeviceSynchronize();
        cudaMemcpy(a.data(), a_d, N * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(what, sizeof what, "%s: option 2 computes B c", order);
        expect(worst(a, Bc, magBc) < 1e-5, what);
        matvecOption3<<<(N + 15) / 16, 16>>>(a_d, B_d, c_d, N);
        cudaDeviceSynchronize();
        cudaMemcpy(a.data(), a_d, N * sizeof(float), cudaMemcpyDeviceToHost);
        std::snprintf(what, sizeof what, "%s: option 3 computes Bt c, not B c", order);
        expect(worst(a, Btc, magBtc) < 1e-5 && worst(a, Bc, magBc) > 1e-2, what);
    }
    cudaFree(B_d);
    cudaFree(c_d);
    cudaFree(a_d);
    return finish();
}
