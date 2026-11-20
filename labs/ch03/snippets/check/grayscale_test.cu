// Runs ../grayscale_kernel.cu, the corrected Fig. 3.4 shown in section 3.2, with its own
// launch, against a CPU reference (within one grey level) and the flat colours lab 3.1 uses.
#include <cstdint>
#include <cstdlib>
#include <vector>
#include "check.h"

void launchColorToGrayscale(unsigned char* Pin_d, unsigned char* Pout_d, int m, int n);

int main() {
    struct Size { int n, m; };   // n rows, m columns, as the launch names them
    const Size sizes[] = {{1, 1}, {1, 300}, {300, 1}, {17, 33}, {62, 76}};
    std::printf("grayscale_kernel.cu\n");
    for (const Size& s : sizes) {
        const int px = s.n * s.m;
        std::vector<unsigned char> rgb(3 * px), gray(px, 0xAB), ref(px);
        uint32_t x = 777u + (uint32_t)px;
        for (unsigned char& v : rgb) { x = x * 1664525u + 1013904223u; v = (unsigned char)(x >> 24); }
        // The first four pixels are pure red, green, blue and white when the picture has room.
        const unsigned char flat[4][3] = {{255, 0, 0}, {0, 255, 0}, {0, 0, 255}, {255, 255, 255}};
        for (int p = 0; p < 4 && p < px; ++p)
            for (int c = 0; c < 3; ++c) rgb[3 * p + c] = flat[p][c];
        for (int p = 0; p < px; ++p)
            ref[p] = (unsigned char)(0.299f * rgb[3 * p] + 0.587f * rgb[3 * p + 1] + 0.114f * rgb[3 * p + 2]);
        unsigned char *in_d = nullptr, *out_d = nullptr;
        cudaMalloc((void**)&in_d, 3 * px);
        cudaMalloc((void**)&out_d, px);
        cudaMemcpy(in_d, rgb.data(), 3 * px, cudaMemcpyHostToDevice);
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            std::vector<unsigned char> got(px, 0xAB);
            cudaMemcpy(out_d, got.data(), px, cudaMemcpyHostToDevice);
            launchColorToGrayscale(in_d, out_d, s.m, s.n);
            cudaDeviceSynchronize();
            cudaMemcpy(got.data(), out_d, px, cudaMemcpyDeviceToHost);
            int off = 0;
            for (int p = 0; p < px; ++p) off += std::abs(int(got[p]) - int(ref[p])) > 1;
            char what[160];
            std::snprintf(what, sizeof what, "%d x %d, %s: every pixel within one level of the CPU reference", s.n, s.m, order);
            expect(off == 0, what);
            if (px >= 4) {
                std::snprintf(what, sizeof what, "%d x %d, %s: red, green, blue, white give 76, 149, 29, 255 (got %d, %d, %d, %d)",
                              s.n, s.m, order, got[0], got[1], got[2], got[3]);
                expect(got[0] == 76 && got[1] == 149 && got[2] == 29 && got[3] == 255, what);
            }
        }
        cudaFree(in_d);
        cudaFree(out_d);
    }
    return finish();
}
