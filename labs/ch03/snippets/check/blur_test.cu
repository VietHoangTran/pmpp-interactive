// Runs ../blur_kernel.cu (Fig. 3.8) against a CPU reference, and ../blur_unsigned.cu
// against blur_kernel.cu, on pictures whose sizes are not multiples of the block.
#include <cstdint>
#include <vector>
#include "check.h"

__global__ void blurKernel(unsigned char *in, unsigned char *out, int w, int h);
__global__ void blurKernelUnsigned(unsigned char *in, unsigned char *out, int w, int h);

// The book's definition, on the host: the average of the part of the 3 x 3 patch that is
// inside the picture, converted to a byte by truncation.
static void cpuBlur(const std::vector<unsigned char>& in, std::vector<unsigned char>& out, int w, int h) {
    for (int row = 0; row < h; ++row)
        for (int col = 0; col < w; ++col) {
            int sum = 0, count = 0;
            for (int r = row - 1; r <= row + 1; ++r)
                for (int c = col - 1; c <= col + 1; ++c)
                    if (r >= 0 && r < h && c >= 0 && c < w) { sum += in[r * w + c]; ++count; }
            out[row * w + col] = (unsigned char)((float)sum / count);
        }
}

int main() {
    struct Size { int h, w; };
    const Size sizes[] = {{1, 1}, {1, 7}, {6, 1}, {7, 6}, {17, 33}, {62, 76}};
    std::printf("blur_kernel.cu and blur_unsigned.cu\n");
    for (const Size& s : sizes) {
        const int n = s.w * s.h;
        std::vector<unsigned char> in(n), ref(n);
        uint32_t x = 12345u + (uint32_t)n;
        for (unsigned char& v : in) { x = x * 1664525u + 1013904223u; v = (unsigned char)(x >> 24); }
        cpuBlur(in, ref, s.w, s.h);
        unsigned char *in_d = nullptr, *a_d = nullptr, *b_d = nullptr;
        cudaMalloc((void**)&in_d, n);
        cudaMalloc((void**)&a_d, n);
        cudaMalloc((void**)&b_d, n);
        cudaMemcpy(in_d, in.data(), n, cudaMemcpyHostToDevice);
        dim3 block(16, 16, 1);
        dim3 grid((s.w + 15) / 16, (s.h + 15) / 16, 1);
        for (int o = 0; o < kOrders; ++o) {
            const char* order = useOrder(o);
            std::vector<unsigned char> a(n, 0xAB), b(n, 0xCD);
            cudaMemcpy(a_d, a.data(), n, cudaMemcpyHostToDevice);
            cudaMemcpy(b_d, b.data(), n, cudaMemcpyHostToDevice);
            blurKernel<<<grid, block>>>(in_d, a_d, s.w, s.h);
            blurKernelUnsigned<<<grid, block>>>(in_d, b_d, s.w, s.h);
            cudaDeviceSynchronize();
            cudaMemcpy(a.data(), a_d, n, cudaMemcpyDeviceToHost);
            cudaMemcpy(b.data(), b_d, n, cudaMemcpyDeviceToHost);
            char what[160];
            std::snprintf(what, sizeof what, "%d x %d, %s: blurKernel matches the CPU reference", s.h, s.w, order);
            expect(a == ref, what);
            std::snprintf(what, sizeof what, "%d x %d, %s: blurKernelUnsigned gives the same output", s.h, s.w, order);
            expect(b == a, what);
        }
        cudaFree(in_d);
        cudaFree(a_d);
        cudaFree(b_d);
    }
    return finish();
}
