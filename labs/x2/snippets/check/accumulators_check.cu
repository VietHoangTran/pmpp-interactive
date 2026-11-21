// Runs the three CUDA fragments of ../accumulators.md (section X2.6, "Safe or unsafe?") on the
// course's CPU emulator, exactly as the page prints them. app/src/sim/qdot.snippets.test.ts
// writes each fragment's fenced code into <fragment id>.inc in a scratch directory, adds that
// directory to the include path, translates this file with labs/tools/cu2emu.py, builds it with
// pmpp/emu/cuda_emu.h force-included, feeds it rows on stdin and compares what it prints with
// app/src/sim/qdot.ts. labs/pmpp.py's CUDA front-end sweep sees this file without the .inc files
// and checks only the driver; the test runs the front-end check with them.
//
// Ours, not in the book. dot4 is the fragments' name for dp4a's arithmetic (accumulators.md):
// here it is __dp4a, which the emulators and the front-end stub now model (CUDA Math API 13.5,
// notes/today/infra-facts.md E2). HIP has no __dp4a (x2-facts B1); this file is never built for
// HIP or run on a GPU.
//
// Input (stdin), whitespace separated, every float as the decimal value of its 32-bit pattern:
//   F <label> <n> <w[0..n-1]> <x[0..n-1]>     float row: dequantised weights, activations
//   I <label> <nb> then per block: 4 qs words, 8 activation words, d4, d8, s8
// Output: one line per fragment run, "<fragment id> <label> <result bits>".
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cuda_fp16.h>

__device__ __forceinline__ int dot4(int a, int b, int c) { return __dp4a(a, b, c); }

#if __has_include("cuda-hfma2.inc")
#define PMPP_X2_FRAGMENTS 1
#endif

// Fragment 2: one __half2 accumulator over the whole row.
__global__ void fragHfma2(const __half2* w2, const __half2* x2, int n, float* out) {
#if defined(PMPP_X2_FRAGMENTS)
#include "cuda-hfma2.inc"
#else
    float y = 0.0f;
    (void)w2; (void)x2; (void)n;
#endif
    *out = y;
}

// Fragments 3 and 4: a block sum in short or int, then float across blocks. Each block gives the
// fragment v (4 words of Q4_0 nibbles), u (8 words of Q8_1 codes), d4, d8 and s8.
__global__ void fragShort(const int* qs, const int* qx, const float* sc, int nb, float* out) {
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const int* v = qs + 4 * b;
        const int* u = qx + 8 * b;
        const float d4 = sc[3 * b], d8 = sc[3 * b + 1], s8 = sc[3 * b + 2];
#if defined(PMPP_X2_FRAGMENTS)
#include "short-sum.inc"
#else
        (void)v; (void)u; (void)d4; (void)d8; (void)s8;
#endif
    }
    *out = sumf;
}
__global__ void fragInt(const int* qs, const int* qx, const float* sc, int nb, float* out) {
    float sumf = 0.0f;
    for (int b = 0; b < nb; ++b) {
        const int* v = qs + 4 * b;
        const int* u = qx + 8 * b;
        const float d4 = sc[3 * b], d8 = sc[3 * b + 1], s8 = sc[3 * b + 2];
#if defined(PMPP_X2_FRAGMENTS)
#include "int-then-float.inc"
#else
        (void)v; (void)u; (void)d4; (void)d8; (void)s8;
#endif
    }
    *out = sumf;
}

static float bitsToFloat(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }
static uint32_t floatBits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }

template <class T> static T* upload(const std::vector<T>& h) {
    T* d = nullptr;
    cudaMalloc((void**)&d, h.size() * sizeof(T) + 1);
    cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice);
    return d;
}
static float runOne(float* out_d) {
    float r = 0.0f;
    cudaMemcpy(&r, out_d, sizeof r, cudaMemcpyDeviceToHost);
    return r;
}

int main() {
    float* out_d = nullptr;
    cudaMalloc((void**)&out_d, sizeof(float));
    char kind[4], label[64];
    int failures = 0;
    while (std::scanf("%3s %63s", kind, label) == 2) {
        if (kind[0] == 'F') {
            int n = 0;
            if (std::scanf("%d", &n) != 1 || n <= 0 || n % 2) return 2;
            std::vector<float> w(n), x(n);
            for (auto* row : {&w, &x})
                for (float& f : *row) { unsigned u; if (std::scanf("%u", &u) != 1) return 2; f = bitsToFloat(u); }
            std::vector<__half2> w2(n / 2), x2(n / 2);
            for (int i = 0; i < n / 2; ++i) {
                w2[i] = __floats2half2_rn(w[2 * i], w[2 * i + 1]);
                x2[i] = __floats2half2_rn(x[2 * i], x[2 * i + 1]);
            }
            __half2* w2_d = upload(w2);
            __half2* x2_d = upload(x2);
            fragHfma2<<<1, 1>>>(w2_d, x2_d, n, out_d);
            if (cudaGetLastError() != cudaSuccess) failures++;
            std::printf("cuda-hfma2 %s %u\n", label, floatBits(runOne(out_d)));
            cudaFree(w2_d);
            cudaFree(x2_d);
        } else if (kind[0] == 'I') {
            int nb = 0;
            if (std::scanf("%d", &nb) != 1 || nb <= 0) return 2;
            std::vector<int> qs(4 * nb), qx(8 * nb);
            std::vector<float> sc(3 * nb);
            for (int b = 0; b < nb; ++b) {
                unsigned u;
                for (int k = 0; k < 4; ++k) { if (std::scanf("%u", &u) != 1) return 2; qs[4 * b + k] = int(u); }
                for (int k = 0; k < 8; ++k) { if (std::scanf("%u", &u) != 1) return 2; qx[8 * b + k] = int(u); }
                for (int k = 0; k < 3; ++k) { if (std::scanf("%u", &u) != 1) return 2; sc[3 * b + k] = bitsToFloat(u); }
            }
            int* qs_d = upload(qs);
            int* qx_d = upload(qx);
            float* sc_d = upload(sc);
            fragShort<<<1, 1>>>(qs_d, qx_d, sc_d, nb, out_d);
            if (cudaGetLastError() != cudaSuccess) failures++;
            std::printf("short-sum %s %u\n", label, floatBits(runOne(out_d)));
            fragInt<<<1, 1>>>(qs_d, qx_d, sc_d, nb, out_d);
            if (cudaGetLastError() != cudaSuccess) failures++;
            std::printf("int-then-float %s %u\n", label, floatBits(runOne(out_d)));
            cudaFree(qs_d);
            cudaFree(qx_d);
            cudaFree(sc_d);
        } else {
            return 2;
        }
    }
    cudaFree(out_d);
    std::printf(failures ? "LAUNCH FAILED\n" : "DONE\n");
    return failures ? 1 : 0;
}
