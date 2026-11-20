// Labs X2.3 and X2.4, provided (do not edit): llama.cpp's Q4_0, Q8_0 and Q8_1 block formats,
// FP16 conversion, and the CPU reference quantisers and dequantisers. Host code only; the
// .cu files include it for the block structs.
//
// Not in the book: module X2 is ours (notes/today/x2-facts.md, "There is no notes/x2.md").
// The layouts and rules are llama.cpp's, read in the facts pass on 2026-11-20 (x2-facts D1,
// D3): ggml/src/ggml-common.h for the structs, ggml/src/ggml-quants.c (the "_ref" functions)
// for Q4_0 and Q8_0, ggml/src/ggml-cuda/quantize.cu for Q8_1.
//
// llama.cpp stores a scale as ggml_half, which is uint16_t in C and C++. Here the scales are
// uint16_t too, holding IEEE binary16 bits, and these functions convert them. The CPU
// emulator does not model __half (appA-facts E2), so the device code converts the bits itself.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

#define QK4_0 32
#define QK8_0 32
#define QK8_1 32

// 18 bytes per 32 weights. Byte j of qs holds weight j in its low nibble and weight j + 16 in
// its high nibble (not two neighbours: x2-facts D1). Weight = (code - 8) * d. Alignment 2.
struct BlockQ4_0 {
    uint16_t d;        // scale, FP16 bits
    uint8_t qs[16];    // 32 codes of 4 bits
};
// 36 bytes per 32 activations (the activation format of llama.cpp's CUDA decode path, D3).
struct BlockQ8_1 {
    uint16_t d;        // scale, FP16 bits: amax / 127
    uint16_t s;        // FP16 bits of d * (sum of the 32 codes), with d before rounding
    int8_t qs[32];     // codes in [-127, 127]
};
static_assert(sizeof(BlockQ4_0) == 18, "block_q4_0 is 18 bytes in llama.cpp");
static_assert(sizeof(BlockQ8_1) == 36, "block_q8_1 is 36 bytes in llama.cpp");

namespace qref {

inline uint32_t floatBits(float f) { uint32_t u; std::memcpy(&u, &f, 4); return u; }
inline float bitsFloat(uint32_t u) { float f; std::memcpy(&f, &u, 4); return f; }

// float -> binary16, round to nearest, ties to even: IEEE 754's default rounding, and what MSL's
// float-to-half conversion does by default (x1-facts C4), which the Metal quantiser of lab X2.3
// relies on. NaN stays NaN.
inline uint16_t floatToHalf(float f) {
    const uint32_t x = floatBits(f);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7FFFFFFFu;
    if (ax > 0x7F800000u) return (uint16_t)(sign | 0x7E00u);          // NaN
    if (ax >= 0x47800000u) return (uint16_t)(sign | 0x7C00u);         // >= 2^16: infinity
    if (ax >= 0x38800000u) {                                          // normal half
        const uint32_t m = ax & 0x7FFFFFu;
        uint32_t h = ((((ax >> 23) - 112u) << 10) | (m >> 13));
        const uint32_t rest = m & 0x1FFFu;
        if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) h += 1;  // a carry may reach infinity: correct
        return (uint16_t)(sign | h);
    }
    if (ax < 0x33000000u) return (uint16_t)sign;                      // below 2^-25: zero
    const uint32_t e = ax >> 23;                                      // 102 .. 112: a subnormal half
    const uint32_t mant = (ax & 0x7FFFFFu) | 0x800000u;
    const uint32_t shift = 126u - e;                                  // 14 .. 24
    uint32_t h = mant >> shift;
    const uint32_t rem = mant & ((1u << shift) - 1u), halfway = 1u << (shift - 1u);
    if (rem > halfway || (rem == halfway && (h & 1u))) h += 1;
    return (uint16_t)(sign | h);
}

// binary16 -> float, exact.
inline float halfToFloat(uint16_t h) {
    const uint32_t sign = (uint32_t)(h & 0x8000u) << 16;
    const uint32_t e = (h >> 10) & 0x1Fu, m = h & 0x3FFu;
    if (e == 0) {
        const float v = (float)m * 5.9604644775390625e-8f;            // m * 2^-24, exact
        return sign ? -v : v;
    }
    if (e == 31) return bitsFloat(sign | 0x7F800000u | (m << 13));
    return bitsFloat(sign | ((e + 112u) << 23) | (m << 13));
}

// ---------------------------------------------------------------- Q4_0
// quantize_row_q4_0_ref (ggml-quants.c, D1): the value of largest magnitude, with its sign,
// becomes code 0 (d = max / -8); codes are x / d + 8.5 truncated, at most 15. n % 32 == 0.
// llama.cpp computes the codes with the float d, then stores d rounded to FP16; so do we.
inline void quantizeQ4_0Row(const float* x, BlockQ4_0* y, int n) {
    for (int i = 0; i < n / QK4_0; ++i) {
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = x[i * QK4_0 + j];
            if (amax < std::fabs(v)) { amax = std::fabs(v); max = v; }
        }
        const float d = max / -8;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = floatToHalf(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const float x0 = x[i * QK4_0 + 0 + j] * id;
            const float x1 = x[i * QK4_0 + QK4_0 / 2 + j] * id;
            const int c0 = (int8_t)(x0 + 8.5f), c1 = (int8_t)(x1 + 8.5f);
            const uint8_t xi0 = (uint8_t)(c0 < 15 ? c0 : 15);
            const uint8_t xi1 = (uint8_t)(c1 < 15 ? c1 : 15);
            y[i].qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}
// dequantize_row_q4_0 (D1): (code - 8) * d, one float multiply per weight.
inline float q4_0Weight(const BlockQ4_0& b, int k) {
    const int code = k < 16 ? (b.qs[k] & 0x0F) : (b.qs[k - 16] >> 4);
    return (float)(code - 8) * halfToFloat(b.d);
}
inline void dequantizeQ4_0Row(const BlockQ4_0* x, float* y, int n) {
    for (int i = 0; i < n / QK4_0; ++i)
        for (int k = 0; k < QK4_0; ++k) y[i * QK4_0 + k] = q4_0Weight(x[i], k);
}

// ---------------------------------------------------------------- Q8_0
// quantize_row_q8_0_ref (D1): d = amax / 127, codes roundf(x / d) in [-127, 127]. The codes
// and the scales are kept in two arrays (lab X2.4's QMatrix layout): q[n], d[n / 32].
inline void quantizeQ8_0Row(const float* x, int8_t* q, uint16_t* dd, int n) {
    for (int i = 0; i < n / QK8_0; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) amax = std::fmax(amax, std::fabs(x[i * QK8_0 + j]));
        const float d = amax / ((1 << 7) - 1);
        const float id = d ? 1.0f / d : 0.0f;
        dd[i] = floatToHalf(d);
        for (int j = 0; j < QK8_0; ++j) q[i * QK8_0 + j] = (int8_t)std::roundf(x[i * QK8_0 + j] * id);
    }
}
inline void dequantizeQ8_0Row(const int8_t* q, const uint16_t* dd, float* y, int n) {
    for (int k = 0; k < n; ++k) y[k] = (float)q[k] * halfToFloat(dd[k / QK8_0]);
}

// ---------------------------------------------------------------- Q8_1
// quantize_q8_1 in llama.cpp's quantize.cu (D3): d = amax / 127; code = roundf(x / d) (0 if
// amax is 0); s = d * (sum of the codes). Both stored as FP16.
inline void quantizeQ8_1Row(const float* x, BlockQ8_1* y, int n) {
    for (int i = 0; i < n / QK8_1; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_1; ++j) amax = std::fmax(amax, std::fabs(x[i * QK8_1 + j]));
        const float d = amax / 127.0f;
        int sum = 0;
        for (int j = 0; j < QK8_1; ++j) {
            const int8_t q = amax == 0.0f ? 0 : (int8_t)std::roundf(x[i * QK8_1 + j] / d);
            y[i].qs[j] = q;
            sum += q;
        }
        y[i].d = floatToHalf(d);
        y[i].s = floatToHalf((float)sum * d);
    }
}

}  // namespace qref
