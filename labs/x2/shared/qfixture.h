// PROVIDED, do not edit. Hand-derived fixtures for the CPU quantisers of labs X2.1 and X2.2:
// for each input, the codes and FP16 scale bits llama.cpp's `_ref` rules give
// (notes/today/x2-facts.md D1), worked out by hand below, not produced by the code they check.
// The lessons' simulator test (app/src/sim/quant.test.ts) is meant to use the same blocks;
// keep the two in step.
//
// Every input is a multiple of a power of two, so every product below is exact in float.
#pragma once
#include <cstdint>

// ---------------------------------------------------------------- FP16 rounding
// IEEE 754 binary16, round to nearest, ties to even.
struct HalfFixture { float value; uint16_t bits; const char* why; };
static const HalfFixture kHalfFixtures[] = {
    {1.0f, 0x3C00, "1 = 2^0: exponent field 15"},
    {-2.0f, 0xC000, "sign bit, exponent field 16"},
    {0.03125f, 0x2800, "2^-5: exponent field 10"},
    {0.125f, 0x3000, "2^-3: exponent field 12"},
    {-0.25f, 0xB400, "-2^-2: sign bit, exponent field 13"},
    {65504.0f, 0x7BFF, "the largest finite FP16"},
    {65520.0f, 0x7C00, "halfway to 65536: ties to even rounds up to infinity"},
    {5.9604644775390625e-8f, 0x0001, "2^-24, the smallest subnormal"},
    {2.98023223876953125e-8f, 0x0000, "2^-25, halfway to the smallest subnormal: ties to even gives 0"},
    {1.00048828125f, 0x3C00, "1 + 2^-11, halfway between 1 and the next FP16: ties to even keeps 1"},
    {1.00146484375f, 0x3C02, "1 + 3 x 2^-11, halfway: ties to even rounds up to 1 + 2^-9"},
    // 1/127 = 0.0078740157... = 2^-7 x 1.007874...; 0.007874 x 1024 = 8.063 fraction steps,
    // nearest 8: exponent field 8, fraction 8.
    {1.0f / 127.0f, 0x2008, "1/127, the Q8_0 scale of a block whose largest magnitude is 1"},
};

// ---------------------------------------------------------------- Q8_0
// One block whose largest magnitude is 127/32, so d = (127/32)/127 = 1/32 exactly
// (FP16 0x2800) and 1/d = 32: each code is roundf(v), v the multiplier of 1/32 below.
// roundf rounds halves away from zero, so 2.5 gives 3 and -2.5 gives -3 (rint would give 2).
static const float kQ8Multipliers[32] = {
    127, -127, 0, 1, -1, 2.5f, -2.5f, 63.5f,
    -63.5f, 0.375f, -0.375f, 0.625f, 100, -100, 50.25f, -50.75f,
    3, 7, 15, 31, 64, -64, 126.5f, -126.5f,
    10, 20, 30, 40, -10, -20, -30, -40,
};
static const int8_t kQ8Codes[32] = {
    127, -127, 0, 1, -1, 3, -3, 64,
    -64, 0, 0, 1, 100, -100, 50, -51,
    3, 7, 15, 31, 64, -64, 127, -127,
    10, 20, 30, 40, -10, -20, -30, -40,
};
static const uint16_t kQ8ScaleBits = 0x2800;   // 1/32
// Weight i of the block is kQ8Multipliers[i] / 32.

// ---------------------------------------------------------------- Q4_0, negative extreme
// Weights u / 8. The first value of largest magnitude is -1 (u = -8; u = +8 comes second and
// loses the tie), so d = -1 / -8 = 1/8 (FP16 0x3000) and 1/d = 8: code = min(15, trunc(u + 8.5)).
static const float kQ4aU[32] = {
    -8, 8, 0, 0.5f, -0.5f, 7, 6.25f, -7.75f,
    -7.25f, 1.5f, -1.5f, 3, -3, 2.25f, -2.25f, 4.5f,
    -4.5f, 5.75f, -5.75f, 0.25f, -0.25f, 1, -1, 2,
    -2, 3.5f, -3.5f, 5, -5, 6, -6, -6.5f,
};
// u + 8.5 = 0.5, 16.5 (clipped to 15), 8.5, 9, 8, 15.5, 14.75, 0.75,
//           1.25, 10, 7, 11.5, 5.5, 10.75, 6.25, 13,
//           4, 14.25, 2.75, 8.75, 8.25, 9.5, 7.5, 10.5,
//           6.5, 12, 5, 13.5, 3.5, 14.5, 2.5, 2
static const uint8_t kQ4aCodes[32] = {
    0, 15, 8, 9, 8, 15, 14, 0,
    1, 10, 7, 11, 5, 10, 6, 13,
    4, 14, 2, 8, 8, 9, 7, 10,
    6, 12, 5, 13, 3, 14, 2, 2,
};
static const uint16_t kQ4aScaleBits = 0x3000;   // +1/8
// Weight i is kQ4aU[i] / 8. Byte j of qs is kQ4aCodes[j] | kQ4aCodes[j + 16] << 4.

// ---------------------------------------------------------------- Q4_0, positive extreme
// Weights w below. The largest magnitude is +2, so d = 2 / -8 = -1/4 (FP16 0xB400, a negative
// scale) and 1/d = -4: code = min(15, trunc(-4w + 8.5)). The second half of the block is zero
// (code 8), so the high nibbles of bytes 8 to 15 are 8.
static const float kQ4bW[32] = {
    2, -2, 0.5f, -0.5f, 1, -1, 0.125f, -0.125f,
    0, 0, 0, 0, 0, 0, 0, 0,
    0.75f, -0.75f, 1.5f, -1.5f, 0.25f, -0.25f, 1.875f, -1.875f,
    0, 0, 0, 0, 0, 0, 0, 0,
};
// -4w + 8.5 = 0.5, 16.5 (clipped to 15), 6.5, 10.5, 4.5, 12.5, 8, 9, then 8.5 eight times,
//             5.5, 11.5, 2.5, 14.5, 7.5, 9.5, 1, 16 (clipped to 15), then 8.5 eight times
static const uint8_t kQ4bCodes[32] = {
    0, 15, 6, 10, 4, 12, 8, 9,
    8, 8, 8, 8, 8, 8, 8, 8,
    5, 11, 2, 14, 7, 9, 1, 15,
    8, 8, 8, 8, 8, 8, 8, 8,
};
static const uint16_t kQ4bScaleBits = 0xB400;   // -1/4

// ---------------------------------------------------------------- Q4_0, all zero
// max = 0, so d = 0 / -8, which IEEE 754 arithmetic makes -0 (it is in JavaScript too):
// FP16 0x8000, negative zero. id = 0 (d compares equal to 0), so every code is
// trunc(0 + 8.5) = 8 and every byte 0x88. Q8_0's all-zero block divides by +127 instead and
// stores +0, 0x0000.
static const uint16_t kQ4ZeroScaleBits = 0x8000;
