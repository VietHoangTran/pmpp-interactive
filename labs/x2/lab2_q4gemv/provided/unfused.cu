// PROVIDED, do not edit. Lab X2.2's unfused baseline (CUDA; also built for HIP and the CPU
// emulators): dequantise the Q4_0 weights into a float32 scratch matrix, then run lab 22.1's
// FP32 gather on it. Two launches, ordered by the default stream.
//   Wf[j][k] = (code - 8) * d   for every weight, then   Y = X Wf^T
// This is the "dequantise, then multiply" route that x2-facts A1 warns about for decode: the
// weights are read as 4.5 bits, then written and read again as 32 bits. llama.cpp uses that
// route only at tile granularity, for prefill (D4). It is here to be measured, for the
// benchmark only. Not in the book; ours.
#include <cstdint>
#include <cstdio>

#include "../../shared/qformat.h"   // BlockQ4_0, QK4_0, halfBitsToFloat

void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 22.1

#define DEQUANT_THREADS 256

// One thread per byte of qs: byte t of block i gives weights 32i + t and 32i + 16 + t.
__global__ void dequantQ4_0Kernel(const BlockQ4_0* W, float* Wf, long long bytes) {
    const long long g = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= bytes) {
        return;
    }
    const long long i = g / (QK4_0 / 2);
    const int t = (int)(g % (QK4_0 / 2));
    const float d = halfBitsToFloat(W[i].d);
    const uint8_t q = W[i].qs[t];
    Wf[i * QK4_0 + t] = (float)((int)(q & 0x0F) - 8) * d;
    Wf[i * QK4_0 + QK4_0 / 2 + t] = (float)((int)(q >> 4) - 8) * d;
}

// Wf is a scratch matrix of out x in floats that the caller owns.
void launchGemvQ4_0Unfused(const float* X, const BlockQ4_0* W, float* Wf, float* Y, int batch, int in, int out) {
    if (batch <= 0 || out <= 0 || in <= 0) {
        return;
    }
    if (in % QK4_0 != 0) {
        std::fprintf(stderr, "launchGemvQ4_0Unfused: in (%d) is not a multiple of %d\n", in, QK4_0);
        return;
    }
    const long long bytes = (long long)out * (in / QK4_0) * (QK4_0 / 2);
    const long long blocks = (bytes + DEQUANT_THREADS - 1) / DEQUANT_THREADS;
    dequantQ4_0Kernel<<<(unsigned int)blocks, DEQUANT_THREADS>>>(W, Wf, bytes);
    launchGemvGather(X, Wf, Y, batch, in, out);   // refuses a batch above 8 itself
}
