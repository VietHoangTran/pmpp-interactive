// PROVIDED, do not edit. Module X2 (not in the book): llama.cpp's Q8_0 and Q4_0 weight
// formats, their reference quantisers and dequantisers, and the FP16 conversions they need,
// as plain loops. Shared by labs X2.1 and X2.2 (owned by that lab group): the drivers name
// these functions, and the CUDA kernels call halfBitsToFloat.
//
// Whose formats these are (notes/today/x2-facts.md D1, checked 2026-11-20 against
// ggml/src/ggml-common.h and ggml/src/ggml-quants.c on llama.cpp master):
//   Q8_0: blocks of QK8_0 = 32 weights, int8 codes in [-127, 127], one FP16 scale d per block,
//         d = amax / 127, code = roundf(w / d)        (quantize_row_q8_0_ref)
//   Q4_0: blocks of QK4_0 = 32 weights, `struct { ggml_half d; uint8_t qs[16]; }`, 18 bytes;
//         d = (the value of largest magnitude, with its sign) / -8, code = min(15, (int8_t)(w / d + 8.5)),
//         byte j holds code j in its low nibble and code j + 16 in its high nibble,
//         value = (code - 8) * d                    (quantize_row_q4_0_ref, dequantize_row_q4_0)
// Both store d as FP16 and compute the codes from the unrounded float d (D1). These are the
// `_ref` rules; llama.cpp's importance-matrix path chooses d differently with the same
// layout (D1), which these labs do not model.
//
// Ours, not llama.cpp's:
//   - Lab X2.1 keeps Q8_0's codes and scales in two arrays (codes [out x in], scales
//     [out x in/32]), as MLX keeps its scales apart from its packed weights (D5), so that the
//     codes stay 4-byte aligned. llama.cpp's block_q8_0 interleaves them (34 bytes, D1).
//   - halfBitsToFloat builds the value with integer operations and a power of two, so that a
//     kernel can convert an FP16 scale held as uint16_t bits with no __half type: the CPU
//     emulators and the front-end stub do not model __half (x2-facts, "Project notes";
//     appA-facts E2), and cuda_fp16.h and hip_fp16.h are different headers (infra B12).
//   - floatToHalfBits rounds to nearest, ties to even, which is IEEE 754's default rounding
//     and what GGML_FP32_TO_FP16 is meant to compute; we did not check llama.cpp's CPU
//     conversion path bit for bit (it depends on the build's F16C support).
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

// __device__ is a macro under nvcc, hipcc, the CPU emulators and the front-end stub, and
// undefined in the drivers' plain C++: the device-side helpers get __host__ __device__ only
// where that means something.
#if defined(__device__)
#define QF_HD __host__ __device__
#else
#define QF_HD
#endif

#define QK8_0 32   // weights per Q8_0 block (llama.cpp's name)
#define QK4_0 32   // weights per Q4_0 block (llama.cpp's name)

// llama.cpp's block_q4_0, with ggml_half spelled as the uint16_t it is in C and C++ (D1).
// 18 bytes, alignment 2: in an array of blocks every other qs starts at an address that is
// 2 mod 4 (A5), so qs is read as bytes or 16-bit words, never as int or char4.
struct BlockQ4_0 {
    uint16_t d;              // FP16 bits of the scale
    uint8_t qs[QK4_0 / 2];   // byte j: code j (low nibble), code j + 16 (high nibble)
};
static_assert(sizeof(BlockQ4_0) == 18, "BlockQ4_0 must be llama.cpp's 18-byte block");
static_assert(alignof(BlockQ4_0) == 2, "BlockQ4_0 is 2-byte aligned, as in llama.cpp");

// 2^n for -24 <= n <= 5, exactly: an integer shift, and for negative n one division by a
// power of two, which is exact in binary floating point.
static inline QF_HD float qfPow2(int n) {
    return n >= 0 ? (float)(1u << n) : 1.0f / (float)(1u << -n);
}

// The float an FP16 bit pattern stands for (IEEE 754 binary16: 1 sign bit, 5 exponent bits
// with bias 15, 10 fraction bits; appA-facts A2). Exact for every pattern: an 11-bit integer
// times a power of two.
static inline QF_HD float halfBitsToFloat(uint16_t h) {
    const uint32_t e = (h >> 10) & 0x1Fu;
    const uint32_t m = h & 0x3FFu;
    float mag;
    if (e == 0) {
        mag = (float)m * qfPow2(-24);                   // zero and subnormals: m * 2^-24
    } else if (e == 31) {
        mag = m ? NAN : INFINITY;                       // infinity, NaN
    } else {
        mag = (float)(m | 0x400u) * qfPow2((int)e - 25);   // (1024 + m) * 2^(e - 15 - 10)
    }
    return (h & 0x8000u) ? -mag : mag;
}

// The FP16 bit pattern nearest to f, ties to even; too large gives infinity, NaN stays NaN.
// Host only (the quantisers run on the CPU).
static inline uint16_t floatToHalfBits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t e = (x >> 23) & 0xFFu;
    uint32_t m = x & 0x7FFFFFu;
    if (e == 0xFFu) {
        return (uint16_t)(sign | 0x7C00u | (m ? 0x200u : 0u));   // infinity, or a quiet NaN
    }
    const int exp = (int)e - 127 + 15;                  // the exponent field FP16 would need
    if (exp >= 31) {
        return (uint16_t)(sign | 0x7C00u);              // beyond FP16's range: infinity
    }
    if (exp <= 0) {                                     // an FP16 subnormal, or zero
        if (exp < -10) {
            return (uint16_t)sign;                      // below half the smallest subnormal
        }
        m |= 0x800000u;                                 // the implicit leading 1
        const int shift = 14 - exp;                     // 13 fraction bits dropped, plus 1 - exp
        uint32_t hm = m >> shift;
        const uint32_t rem = m & ((1u << shift) - 1u), halfway = 1u << (shift - 1);
        if (rem > halfway || (rem == halfway && (hm & 1u))) {
            ++hm;                                       // may carry into the exponent: still right
        }
        return (uint16_t)(sign | hm);
    }
    uint32_t h = sign | ((uint32_t)exp << 10) | (m >> 13);
    const uint32_t rem = m & 0x1FFFu;
    if (rem > 0x1000u || (rem == 0x1000u && (h & 1u))) {
        ++h;                                            // a carry may reach infinity: still right
    }
    return (uint16_t)h;
}

// ---------------------------------------------------------------- Q8_0
// quantize_row_q8_0_ref (D1) for one row of n weights, n a multiple of QK8_0, with codes and
// scales written to two arrays: q[k] for weight k, d[k / QK8_0] for its block.
static inline void quantizeRowQ8_0Ref(const float* x, int8_t* q, uint16_t* d, int n) {
    for (int i = 0; i < n / QK8_0; ++i) {
        float amax = 0.0f;
        for (int j = 0; j < QK8_0; ++j) {
            amax = std::fmax(amax, std::fabs(x[i * QK8_0 + j]));
        }
        const float dd = amax / ((1 << 7) - 1);
        const float id = dd ? 1.0f / dd : 0.0f;
        d[i] = floatToHalfBits(dd);
        for (int j = 0; j < QK8_0; ++j) {
            q[i * QK8_0 + j] = (int8_t)std::roundf(x[i * QK8_0 + j] * id);   // half away from zero
        }
    }
}

// dequantize_row_q8_0 (D1): w = code * FP16(d).
static inline void dequantizeRowQ8_0(const int8_t* q, const uint16_t* d, float* y, int n) {
    for (int k = 0; k < n; ++k) {
        y[k] = q[k] * halfBitsToFloat(d[k / QK8_0]);
    }
}

// ---------------------------------------------------------------- Q4_0
// quantize_row_q4_0_ref (D1) for one row of n weights, n a multiple of QK4_0.
static inline void quantizeRowQ4_0Ref(const float* x, BlockQ4_0* y, int n) {
    for (int i = 0; i < n / QK4_0; ++i) {
        const float* xb = x + i * QK4_0;
        float amax = 0.0f, max = 0.0f;
        for (int j = 0; j < QK4_0; ++j) {
            const float v = xb[j];
            if (amax < std::fabs(v)) {   // strictly less: the first of two equal magnitudes wins
                amax = std::fabs(v);
                max = v;
            }
        }
        const float d = max / -8;
        const float id = d ? 1.0f / d : 0.0f;
        y[i].d = floatToHalfBits(d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const float x0 = xb[j] * id;
            const float x1 = xb[QK4_0 / 2 + j] * id;
            const int c0 = (int8_t)(x0 + 8.5f), c1 = (int8_t)(x1 + 8.5f);   // truncation toward zero
            const uint8_t xi0 = (uint8_t)(c0 < 15 ? c0 : 15);
            const uint8_t xi1 = (uint8_t)(c1 < 15 ? c1 : 15);
            y[i].qs[j] = (uint8_t)(xi0 | (xi1 << 4));
        }
    }
}

// dequantize_row_q4_0 (D1): w = (code - 8) * FP16(d), weight j from the low nibble of byte j
// and weight j + 16 from its high nibble.
static inline void dequantizeRowQ4_0(const BlockQ4_0* x, float* y, int n) {
    for (int i = 0; i < n / QK4_0; ++i) {
        const float d = halfBitsToFloat(x[i].d);
        for (int j = 0; j < QK4_0 / 2; ++j) {
            const int x0 = (x[i].qs[j] & 0x0F) - 8;
            const int x1 = (x[i].qs[j] >> 4) - 8;
            y[i * QK4_0 + j] = x0 * d;
            y[i * QK4_0 + j + QK4_0 / 2] = x1 * d;
        }
    }
}
