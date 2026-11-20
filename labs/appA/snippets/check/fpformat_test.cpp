// Cross-check for lab A.1's ../../lab1_sumerror/fpformat.h. The unit test
// app/src/sim/minifloat.test.ts compiles this file, runs it and compares every line, bit for
// bit, with the model behind sections A.1 and A.2 (app/src/sim/minifloat.ts).
//
//   fpformat_test        prints one line per case:
//     r <fmt> <mode> <x as 16 hex digits of a double> <stored bits, hex>
//     a <fmt> <mode> <a bits, hex> <b bits, hex> <bits of a + b, hex>
//   fmt is fp32, fp16 or bf16; mode is rne (to nearest, ties to even) or rtz (toward zero).
//
// Cases: random doubles from far below the smallest subnormal to far above the largest value;
// the exact midpoint between two neighbouring values of the format, its negation, and the
// doubles on either side of it; the overflow threshold; and additions of two random values
// of the format, which for FP32 and BF16 include large exponent gaps. Infinite inputs are
// left out: both sides keep them infinite in every mode, as IEEE 754 does (only a finite
// overflow goes to the largest finite value toward zero), and the lab only rounds finite sums.
#include "../../lab1_sumerror/fpformat.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>

using namespace fpformat;

static std::uint64_t state = 0x9E3779B97F4A7C15ull;
static std::uint64_t next() {  // xorshift64: deterministic, so a failure reproduces
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}
static std::uint64_t bitsOfDouble(double d) {
    std::uint64_t u;
    std::memcpy(&u, &d, sizeof u);
    return u;
}
static const char* modeTag(Mode m) { return m == Mode::Nearest ? "rne" : "rtz"; }

static void roundCase(const Format& f, Mode mode, double x) {
    std::printf("r %s %s %016llx %x\n", f.name, modeTag(mode), (unsigned long long)bitsOfDouble(x), (unsigned)encode(f, x, mode));
}

int main() {
    const Format formats[] = {FP32, FP16, BF16};
    const Mode modes[] = {Mode::Nearest, Mode::TowardZero};
    for (const Format& f : formats)
        for (Mode mode : modes) {
            const std::uint64_t positiveCodes = 1ull << (f.e + f.m);
            const std::uint64_t finitePositive = positiveCodes - (1ull << f.m);  // below the all-ones exponent
            // Random doubles over the whole range and beyond it.
            for (int i = 0; i < 3000; ++i) {
                const std::uint64_t r = next();
                const int k = (int)(r % 300) - 160;
                const double mant = 1.0 + (double)(next() >> 12) / 4503599627370496.0;  // [1, 2), 52 random bits
                const double x = std::ldexp(mant, k);
                roundCase(f, mode, (r >> 40) & 1 ? -x : x);
            }
            // Exact ties between neighbours, and the doubles just either side of them.
            for (int i = 0; i < 1500; ++i) {
                const std::uint32_t c = (std::uint32_t)(next() % (finitePositive - 1));
                const double mid = (decode(f, c) + decode(f, c + 1)) / 2;  // exact in double
                roundCase(f, mode, mid);
                roundCase(f, mode, -mid);
                roundCase(f, mode, std::nextafter(mid, 0.0));
                roundCase(f, mode, std::nextafter(mid, INFINITY));
            }
            // The overflow threshold: half a spacing above the largest value.
            const double edge = maxFinite(f) + ulp(f, maxFinite(f)) / 2;
            roundCase(f, mode, edge);
            roundCase(f, mode, std::nextafter(edge, 0.0));
            roundCase(f, mode, -edge);
            // a + b for two finite values of the format, either sign.
            for (int i = 0; i < 3000; ++i) {
                const std::uint32_t a = (std::uint32_t)(next() % (2 * positiveCodes));
                const std::uint32_t b = (std::uint32_t)(next() % (2 * positiveCodes));
                const double va = decode(f, a);
                const double vb = decode(f, b);
                if (!std::isfinite(va) || !std::isfinite(vb)) continue;
                std::printf("a %s %s %x %x %x\n", f.name, modeTag(mode), (unsigned)a, (unsigned)b,
                            (unsigned)bitsOf(f, add(f, va, vb, mode)));
            }
        }
    return 0;
}
