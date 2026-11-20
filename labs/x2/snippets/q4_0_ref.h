// One llama.cpp Q4_0 block on the CPU: quantise, pack, dequantise, and the block's dot
// product in the folded form the GPU kernels use. Plain host C++, ours, written from the
// reference rule llama.cpp uses to create model files (quantize_row_q4_0_ref and
// dequantize_row_q4_0 in ggml/src/ggml-quants.c, quoted in notes/today/x2-facts.md D1).
// The FP16 conversion is our own routine (round to nearest even). Nothing here runs on a GPU.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

constexpr int QK4_0 = 32;

struct BlockQ4_0 {
    uint16_t d;              // the scale, as FP16 bits (llama.cpp's ggml_half)
    uint8_t qs[QK4_0 / 2];   // byte j: weight j in the low nibble, weight j + 16 in the high one
};
static_assert(sizeof(BlockQ4_0) == 18, "llama.cpp's block_q4_0 is 18 bytes");
static_assert(alignof(BlockQ4_0) == 2, "and only 2-byte aligned");

inline uint16_t fp32_to_fp16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const uint32_t ax = x & 0x7FFFFFFFu;
    if (ax >= 0x7F800000u) return sign | (ax > 0x7F800000u ? 0x7E00u : 0x7C00u);  // NaN, infinity
    if (ax >= 0x477FF000u) return sign | 0x7C00u;  // 65520 and above round to infinity
    if (ax < 0x38800000u) {                         // below 2^-14: an FP16 subnormal or zero
        float a;
        std::memcpy(&a, &ax, 4);
        return sign | static_cast<uint16_t>(std::nearbyint(a * 16777216.0f));  // units of 2^-24
    }
    uint32_t h = (ax >> 13) - ((127u - 15u) << 10);  // rebias the exponent, keep 10 mantissa bits
    const uint32_t rest = ax & 0x1FFFu;              // the 13 bits dropped
    if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;
    return static_cast<uint16_t>(sign | h);
}

inline float fp16_bits_to_fp32(uint16_t h) {
    const float s = (h & 0x8000u) ? -1.0f : 1.0f;
    const int e = (h >> 10) & 0x1F;
    const int m = h & 0x3FF;
    if (e == 0) return s * std::ldexp(static_cast<float>(m), -24);
    if (e == 31) return m ? NAN : s * INFINITY;
    return s * std::ldexp(static_cast<float>(1024 + m), e - 25);
}

// The reference rule: the scale is the value of largest magnitude, sign included, over -8.
inline void quantize_q4_0_block(const float* x, BlockQ4_0* y) {
    float amax = 0.0f;  // the largest magnitude
    float max = 0.0f;   // the value that has it
    for (int j = 0; j < QK4_0; ++j) {
        const float v = x[j];
        if (amax < std::fabs(v)) {
            amax = std::fabs(v);
            max = v;
        }
    }
    const float d = max / -8;  // the extreme lands exactly on code 0
    const float id = d ? 1.0f / d : 0.0f;
    y->d = fp32_to_fp16_bits(d);  // codes come from the float d; the block keeps FP16(d)
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const float x0 = x[j] * id;
        const float x1 = x[QK4_0 / 2 + j] * id;
        const int t0 = static_cast<int8_t>(x0 + 8.5f);  // add 8.5, truncate
        const int t1 = static_cast<int8_t>(x1 + 8.5f);
        const uint8_t xi0 = static_cast<uint8_t>(t0 < 15 ? t0 : 15);  // +8 clips to +7
        const uint8_t xi1 = static_cast<uint8_t>(t1 < 15 ? t1 : 15);
        y->qs[j] = static_cast<uint8_t>(xi0 | (xi1 << 4));
    }
}

inline void dequantize_q4_0_block(const BlockQ4_0* x, float* y) {
    const float d = fp16_bits_to_fp32(x->d);
    for (int j = 0; j < QK4_0 / 2; ++j) {
        const int x0 = (x->qs[j] & 0x0F) - 8;
        const int x1 = (x->qs[j] >> 4) - 8;
        y[j] = x0 * d;
        y[j + QK4_0 / 2] = x1 * d;
    }
}

// Word i (0..3) of the quants, built from two 16-bit reads as llama.cpp's get_int_b2 does:
// in an array of 18-byte blocks, every other block's qs starts 2 bytes past a 4-byte boundary.
inline uint32_t q4_0_word(const BlockQ4_0* b, int i) {
    uint16_t lo, hi;
    std::memcpy(&lo, b->qs + 4 * i, 2);
    std::memcpy(&hi, b->qs + 4 * i + 2, 2);
    return static_cast<uint32_t>(lo) | (static_cast<uint32_t>(hi) << 16);
}

// Sum over the block of dequantised weight times activation, with the 8 taken out of the
// loop: d * sum(q * y) - 8 * d * sum(y).
inline float dot_q4_0_block(const BlockQ4_0* b, const float* y) {
    float sumqy = 0.0f;
    float sumy = 0.0f;
    for (int j = 0; j < QK4_0; ++j) {
        const int q = j < QK4_0 / 2 ? (b->qs[j] & 0x0F) : (b->qs[j - QK4_0 / 2] >> 4);
        sumqy += q * y[j];
        sumy += y[j];
    }
    const float d = fp16_bits_to_fp32(b->d);
    return d * sumqy - 8 * d * sumy;
}
