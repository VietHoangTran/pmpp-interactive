// Lab X2.3, provided (do not edit): the block formats and the functions kernel.cu defines.
// Not in the book: module X2 is ours. The formats are llama.cpp's (notes/today/x2-facts.md D1).
#pragma once
#include <cstdint>

#include "../shared-g12/qformats.h"   // BlockQ4_0, BlockQ8_1, and the CPU reference quantisers

// The GEMV keeps one running sum per batch row in registers, as lab 22.1's gather does; llama.cpp's
// matrix-vector kernels stop at 8 tokens too (x2-facts, "Already settled": ch15 D1, ch22 D3).
#define QHUNT_MAX_BATCH 8

#if !defined(__OBJC__)
// ---- provided and correct, in <variant>/kernel.cu ----
// Xq[b * in / 32 + k] = the Q8_1 block of X[b][32k .. 32k + 31]. in % 32 == 0.
void launchQuantizeQ8_1(const float* X, BlockQ8_1* Xq, int batch, int in);
// ---- the bug hunt, in <variant>/kernel.cu ----
// Y[batch x out] = X W^T on the integer path: W is (out, in) as out * in / 32 Q4_0 blocks,
// Xq the Q8_1 activations. batch <= QHUNT_MAX_BATCH, in % 32 == 0.
void launchGemvQ4_0Q8_1(const BlockQ4_0* W, const BlockQ8_1* Xq, float* Y, int batch, int in, int out);
#endif
