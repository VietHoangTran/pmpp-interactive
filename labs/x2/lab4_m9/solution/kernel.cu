// Lab X2.4 reference solution (CUDA; also built for HIP and for the CPU emulators).
// Milestone M9: lab 22.2's greedy decode step (M8), with its weights stored in llama.cpp's
// Q8_0 or Q4_0 formats, so that the benchmark can measure what fewer bytes per weight buy.
//
// Not in the book: module X2 is ours. Its anchor is appendix A section A.5 (p. 593, recorded
// in notes/appA.md): a smaller format moves fewer bytes per operation, which raises the
// intensity of a bandwidth-bound kernel; decode is one (ch22-facts A5). The formats and their
// rules are llama.cpp's (notes/today/x2-facts.md D1); the step is lab 22.2's.
//
// The step, for s.batch sequences at once (qdecode.h has the shapes):
//   H = dequantise(E)[tokens]              embed: output-centric gather, reads its block's scale
//   for each layer l:
//     T = H W_l^T                          launchQLinear: the GEMV for W_l's format
//     T += b_l                             M0 (lab 2.3), in place
//     H += T                               M0
//   logits = H E^T                         launchQLinear with E (tied to the embedding)
//   probs = softmax(logits / temperature)  M4 (lab 10.3)
//   tokens = argmax(logits)                M4, over the tokens this step read
// One launch per operation for the whole batch, and nothing waits for the GPU: launches with
// no stream run in issue order (ch02-facts A5, B3).
//
// The weights stay quantised in device memory; nothing here writes a dequantised copy. That is
// the point (x2-facts A1): an unfused "dequantise, then multiply" moves more bytes than the
// float32 step it replaces. The GEMVs dequantise inside their dot products.
#include <cstdio>

#include "../qdecode.h"

const int threadsPerBlock = 256;   // the embedding's blocks, as M0 uses

// FP16 bits to float, exact. The emulators model no __half (appA-facts E2), so the bits are
// converted by hand, as qref::halfToFloat does on the CPU; the result is the same float.
static __device__ float halfToFloat(uint16_t h) {
    const unsigned sign = (unsigned)(h & 0x8000u) << 16;
    const unsigned e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f;   // subnormal: m * 2^-24
        return sign ? -v : v;
    }
    union { unsigned u; float f; } c;
    c.u = (e == 31) ? (sign | 0x7F800000u | (m << 13)) : (sign | ((e + 112u) << 23) | (m << 13));
    return c.f;
}

// ---------------------------------------------------------------- the embedding
// Lab 22.2's gather, unchanged: thread i owns H[i] = E[tokens[r]][j].
__global__ void embedRowsF32Kernel(const float* E, const int* tokens, float* H, int batch, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * d) {
        int r = i / d, j = i % d;
        H[i] = E[(size_t)tokens[r] * d + j];
    }
}

// Q8_0: weight = code * scale. Element j of row t is code q8[t * d + j] and scale
// q8d[t * (d / 32) + j / 32]. Neighbouring threads read neighbouring codes and, 32 at a time,
// the same scale.
__global__ void embedRowsQ8_0Kernel(const int8_t* q8, const uint16_t* q8d, const int* tokens, float* H, int batch,
                                    int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * d) {
        int r = i / d, j = i % d;
        const size_t t = (size_t)tokens[r];
        H[i] = (float)q8[t * d + j] * halfToFloat(q8d[t * (d / QK8_0) + j / QK8_0]);
    }
}

// Q4_0: weight = (code - 8) * scale. Element j of row t is in block t * (d / 32) + j / 32, at
// position k = j % 32 of that block: the low nibble of byte k for k < 16, the high nibble of
// byte k - 16 otherwise (llama.cpp's j / j + 16 order, x2-facts D1).
__global__ void embedRowsQ4_0Kernel(const BlockQ4_0* q4, const int* tokens, float* H, int batch, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * d) {
        int r = i / d, j = i % d;
        const BlockQ4_0& blk = q4[(size_t)tokens[r] * (d / QK4_0) + j / QK4_0];
        const int k = j % QK4_0;
        const int code = k < 16 ? (blk.qs[k] & 0x0F) : (blk.qs[k - 16] >> 4);
        H[i] = (float)(code - 8) * halfToFloat(blk.d);
    }
}

void launchEmbedQ(const QMatrix& E, const int* tokens, float* H, int batch) {
    const int d = E.cols;
    const int n = batch * d;
    if (n <= 0) return;   // an empty grid is not a valid launch
    const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;
    switch (E.fmt) {
        case QF_F32: embedRowsF32Kernel<<<blocks, threadsPerBlock>>>(E.f32, tokens, H, batch, d); break;
        case QF_Q8_0: embedRowsQ8_0Kernel<<<blocks, threadsPerBlock>>>(E.q8, E.q8d, tokens, H, batch, d); break;
        case QF_Q4_0: embedRowsQ4_0Kernel<<<blocks, threadsPerBlock>>>(E.q4, tokens, H, batch, d); break;
    }
}

// ---------------------------------------------------------------- the linear layer
void launchQLinear(const float* X, const QMatrix& W, float* Y, int batch) {
    if (batch > QDECODE_MAX_BATCH) {
        std::fprintf(stderr, "launchQLinear: batch %d is above %d (the GEMVs are matrix-vector kernels)\n", batch,
                     QDECODE_MAX_BATCH);
        return;
    }
    // (out, in) = (rows, cols): the same layout every GEMV of the project takes.
    switch (W.fmt) {
        case QF_F32: launchGemvGather(X, W.f32, Y, batch, W.cols, W.rows); break;
        case QF_Q8_0: launchGemvQ8_0(X, W.q8, W.q8d, Y, batch, W.cols, W.rows); break;
        case QF_Q4_0: launchGemvQ4_0(X, W.q4, Y, batch, W.cols, W.rows); break;
    }
}

// ---------------------------------------------------------------- the step
void launchQDecodeStep(const QDecodeModel& m, DecodeState& s) {
    launchEmbedQ(m.E, s.tokens, s.H, s.batch);
    for (int l = 0; l < m.layers; ++l) {
        launchQLinear(s.H, m.W[l], s.T, s.batch);              // T = H W_l^T, in W_l's format
        launchAddBias(s.T, s.T, m.b[l], s.batch, m.d);         // T += b_l
        launchAddResidual(s.H, s.T, s.batch * m.d);            // H += T
    }
    launchQLinear(s.H, m.E, s.logits, s.batch);                // the LM head: E is (vocab, d)
    launchSoftmaxRows(s.logits, s.probs, s.batch, m.vocab, 1.0f / s.temperature);
    launchArgmaxRows(s.logits, s.tokens, s.batch, m.vocab);    // the next step starts on the device
}
