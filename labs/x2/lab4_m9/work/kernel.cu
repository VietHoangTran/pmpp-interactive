// Lab X2.4 starter (CUDA; also built for HIP and for the CPU emulators).
// Milestone M9: lab 22.2's greedy decode step (M8), with its weights stored in llama.cpp's
// Q8_0 or Q4_0 formats, so that the benchmark can measure what fewer bytes per weight buy.
//
// Not in the book: module X2 is ours. Its anchor is appendix A section A.5 (p. 593, recorded
// in notes/appA.md): a smaller format moves fewer bytes per operation, which raises the
// intensity of a bandwidth-bound kernel; decode is one (ch22-facts A5). The formats and their
// rules are llama.cpp's (notes/today/x2-facts.md D1); the step is lab 22.2's.
//
// qdecode.h has the formats, QMatrix, QDecodeModel, DecodeState, and the earlier labs' pieces
// you call. ../shared-g12/qformats.h has BlockQ4_0 and the CPU's dequantisers, which your
// embedding must match bit for bit.
#include <cstdio>

#include "../qdecode.h"

const int threadsPerBlock = 256;   // the embedding's blocks, as M0 uses

// Provided. FP16 bits to float, exact. The emulators model no __half (appA-facts E2), so the
// bits are converted by hand, as qref::halfToFloat does on the CPU; the result is the same float.
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
// Provided: lab 22.2's gather, unchanged. Thread i owns H[i] = E[tokens[r]][j].
__global__ void embedRowsF32Kernel(const float* E, const int* tokens, float* H, int batch, int d) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < batch * d) {
        int r = i / d, j = i % d;
        H[i] = E[(size_t)tokens[r] * d + j];
    }
}

// TODO 1: the same gather for a Q8_0 table. Row t of E is d int8 codes q8[t * d .. t * d + d - 1]
//         and d / 32 FP16 scales in q8d, one per 32 codes of the row. Which scale belongs to
//         element j of row t? One float multiply per element: compute it the way
//         qref::dequantizeQ8_0Row does, or the bit-for-bit test will tell you.
__global__ void embedRowsQ8_0Kernel(const int8_t* q8, const uint16_t* q8d, const int* tokens, float* H, int batch,
                                    int d) {
}

// TODO 2: the same gather for a Q4_0 table: row t of E is d / 32 BlockQ4_0 blocks. Element j
//         is in which block, at which position k of it, and which nibble of which byte holds
//         position k? (qformats.h says how llama.cpp packs a block; it is not two neighbours
//         per byte.) Then weight = (code - 8) * scale, as qref::q4_0Weight computes it.
__global__ void embedRowsQ4_0Kernel(const BlockQ4_0* q4, const int* tokens, float* H, int batch, int d) {
}

void launchEmbedQ(const QMatrix& E, const int* tokens, float* H, int batch) {
    // TODO 3: one thread per element of H (batch * E.cols of them), blocks of threadsPerBlock;
    //         launch the kernel for E.fmt with the pointers that format uses. An empty grid is
    //         not a valid launch.
}

// ---------------------------------------------------------------- the linear layer
void launchQLinear(const float* X, const QMatrix& W, float* Y, int batch) {
    // TODO 4: Y = X W^T, by W.fmt: lab 22.1's launchGemvGather, lab X2.1's launchGemvQ8_0 or
    //         lab X2.2's launchGemvQ4_0 (qdecode.h has their argument orders). W is (out, in):
    //         which of W.rows and W.cols is `in`? Refuse a batch above QDECODE_MAX_BATCH with a
    //         message: these are matrix-vector kernels.
}

// ---------------------------------------------------------------- the step
void launchQDecodeStep(const QDecodeModel& m, DecodeState& s) {
    // TODO 5: lab 22.2's step, with your two functions in place of its embedding and its
    //         linear layer. One launch per operation for the whole batch, nothing that waits:
    //           H = dequantised E[tokens];
    //           for each layer l: T = H W_l^T, T += b_l, H += T;
    //           logits = H E^T (E is the LM head's weight too);
    //           probs = softmax(logits) with scale 1 / s.temperature;
    //           the next tokens = argmax(logits), where the next step will read them.
}
