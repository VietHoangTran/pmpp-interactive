// One MXFP4 block (OCP MX v1.1, ours: not in the book). 32 weights share one
// power-of-two scale X, stored as a UE8M0 byte; each weight is a 4-bit E2M1 code.
#pragma once
#include <cmath>
#include <cstdint>

constexpr int kBlock = 32;
// The E2M1 magnitudes in code order: the 3-bit code of a magnitude is its index.
constexpr float kE2M1[8] = {0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f};
constexpr int kE2M1MaxExp = 2;  // the largest value, 6, is 1.5 x 2^2

struct MxBlock {
    std::uint8_t scale;              // UE8M0: X = 2^(scale - 127)
    std::uint8_t codes[kBlock / 2];  // two codes per byte, even elements in the low nibble
};

// The nearest E2M1 code to x, which is already divided by X. Past 6 it clamps;
// on a tie it takes the even code, which is round to nearest, ties to even.
inline std::uint8_t e2m1_code(float x) {
    float a = std::fmin(std::fabs(x), 6.0f);
    int best = 0;
    for (int c = 1; c < 8; ++c) {
        float d = std::fabs(a - kE2M1[c]);
        float b = std::fabs(a - kE2M1[best]);
        if (d < b || (d == b && c % 2 == 0)) best = c;
    }
    return static_cast<std::uint8_t>(best | (std::signbit(x) ? 8 : 0));
}

inline MxBlock mx_quantize(const float* v) {
    // 1. A max reduction: the largest magnitude in the block.
    float amax = 0.0f;
    for (int i = 0; i < kBlock; ++i) amax = std::fmax(amax, std::fabs(v[i]));
    // 2. The shared exponent: floor(log2 amax) minus E2M1's largest exponent,
    //    kept inside UE8M0's range of -127 to 127.
    int e = amax > 0.0f ? std::ilogb(amax) - kE2M1MaxExp : -127;
    e = e < -127 ? -127 : (e > 127 ? 127 : e);
    MxBlock out{};
    out.scale = static_cast<std::uint8_t>(e + 127);
    // 3. Divide by X = 2^e, snap to the E2M1 grid, pack two codes per byte.
    for (int i = 0; i < kBlock; ++i) {
        std::uint8_t c = e2m1_code(std::ldexp(v[i], -e));
        out.codes[i / 2] |= static_cast<std::uint8_t>(i % 2 == 0 ? c : c << 4);
    }
    return out;
}

// What a kernel reads back for element i: X times the element's value.
inline float mx_decode(const MxBlock& b, int i) {
    int c = (i % 2 == 0) ? (b.codes[i / 2] & 0xF) : (b.codes[i / 2] >> 4);
    float m = kE2M1[c & 7];
    return std::ldexp((c & 8) ? -m : m, b.scale - 127);
}
