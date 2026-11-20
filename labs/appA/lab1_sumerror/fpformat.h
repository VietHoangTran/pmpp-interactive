// Lab A.1, provided (do not edit): FP32, FP16 and BF16 in software, so the driver can say
// what a sum in each format must return, bit for bit.
//
// Everything is computed in double. Every value of the three formats is a double, so
// decoding is exact; rounding is exact too (see roundPair). The rules are the ones of
// app/src/sim/minifloat.ts (the explorer of sections A.1 to A.4): round to nearest with
// ties to even, or toward zero; an overflow gives infinity to nearest and the largest finite
// value toward zero (IEEE 754 §7.4); subnormals are kept. A device that flushes subnormals
// differs from this model only where a result lands below the smallest normal value.
//
// Formats (notes/today/appA-facts.md A2): FP32 1-8-23, bias 127; FP16 1-5-10, bias 15;
// BF16 1-8-7, bias 127. All three reserve the all-ones exponent for infinity and NaN.
// FP16 and BF16 are not in the book.
#pragma once
#include <cmath>
#include <cstdint>
#include <limits>

namespace fpformat {

struct Format {
    const char* name;
    int e;      // exponent bits
    int m;      // stored mantissa bits (the hidden 1 not counted)
    int bias;
};
constexpr Format FP32{"fp32", 8, 23, 127};
constexpr Format FP16{"fp16", 5, 10, 15};
constexpr Format BF16{"bf16", 8, 7, 127};

enum class Mode { Nearest, TowardZero };
inline const char* modeName(Mode r) { return r == Mode::Nearest ? "nearest" : "toward-zero"; }

inline int precision(const Format& f) { return f.m + 1; }              // significant bits, p
inline int emin(const Format& f) { return 1 - f.bias; }                 // binade of the smallest normal
inline int emax(const Format& f) { return (1 << f.e) - 2 - f.bias; }    // binade of the largest finite
inline double maxFinite(const Format& f) { return std::ldexp(2.0 - std::ldexp(1.0, -f.m), emax(f)); }
inline double minNormal(const Format& f) { return std::ldexp(1.0, emin(f)); }
// The unit roundoff u = 2^-p: the largest relative error of one rounding to nearest.
// Rounding toward zero can be wrong by almost 2u.
inline double unitRoundoff(const Format& f) { return std::ldexp(1.0, -precision(f)); }

// The spacing of the format's values at x: 2^(k - m) in the binade [2^k, 2^(k+1)), the
// subnormal spacing below the smallest normal, the top binade's spacing above the largest
// value. As minifloat.ts's ulp().
inline double ulp(const Format& f, double x) {
    const double a = std::fabs(x);
    int k = (a == 0 || !std::isfinite(a)) ? emin(f) : std::ilogb(a);
    if (k < emin(f)) k = emin(f);
    if (k > emax(f)) k = emax(f);
    return std::ldexp(1.0, k - f.m);
}

// The value the format stores for x + tail, where x is a double and tail is the exact
// remainder of the operation that produced x (0 when there is none). |tail| is at most half
// a unit in the last place of x as a double, which is far below the format's spacing.
inline double roundPair(const Format& f, double x, double tail, Mode mode) {
    if (std::isnan(x)) return x;
    const bool neg = std::signbit(x);
    if (std::isinf(x)) return x;                         // an infinite input stays infinite in every mode (IEEE 754)
    if (x == 0) return x;
    const double a = std::fabs(x);
    const double t = neg ? -tail : tail;                 // > 0: the exact value is further from 0
    const int k = std::ilogb(a);
    if (k > emax(f)) {                                   // beyond the top binade
        const double big = mode == Mode::TowardZero ? maxFinite(f) : std::numeric_limits<double>::infinity();
        return neg ? -big : big;
    }
    const int q = k < emin(f) ? emin(f) : k;              // subnormals share the lowest spacing
    const double quantum = std::ldexp(1.0, q - f.m);
    const double scaled = a / quantum;                   // exact: a power-of-two scaling
    const double fl = std::floor(scaled);
    const double frac = scaled - fl;                     // exact
    double r;
    if (mode == Mode::TowardZero) {
        if (frac == 0 && t < 0) {
            // a is on the grid but the exact value is just below it: the next value down.
            // Below a power of two the spacing halves (except at the subnormal boundary).
            const bool powerOfTwo = fl == std::ldexp(1.0, f.m) && q > emin(f);
            r = a - (powerOfTwo ? quantum / 2 : quantum);
        } else {
            r = fl * quantum;
        }
    } else {
        bool up;
        if (frac > 0.5) up = true;
        else if (frac < 0.5) up = false;
        else if (t != 0) up = t > 0;                     // not really a tie
        else up = std::fmod(fl, 2.0) != 0;               // a tie: to the even neighbour
        r = (up ? fl + 1 : fl) * quantum;
    }
    if (r > maxFinite(f)) r = mode == Mode::TowardZero ? maxFinite(f) : std::numeric_limits<double>::infinity();
    return neg ? -r : r;
}

// The value the format stores for x.
inline double round(const Format& f, double x, Mode mode = Mode::Nearest) { return roundPair(f, x, 0, mode); }

// a + b in the format, rounded once. a and b must be values of the format. The double sum s
// may itself be rounded (a large exponent gap); TwoSum recovers the exact remainder, so the
// result is right in both modes. (Rounding s alone would be wrong toward zero for, say,
// 1 + (-2^-100).) The host compiler must not reassociate this: Clang and GCC do not by
// default (appA-facts B5).
inline double add(const Format& f, double a, double b, Mode mode) {
    const double s = a + b;
    if (!std::isfinite(s)) return roundPair(f, s, 0, mode);
    if (s == 0) return s;
    const double bb = s - a;
    const double tail = (a - (s - bb)) + (b - bb);
    return roundPair(f, s, tail, mode);
}
inline double sub(const Format& f, double a, double b, Mode mode) { return add(f, a, -b, mode); }

// The bit pattern of a value of the format (v must be representable, or infinite, or NaN).
// NaN is the positive quiet NaN, as in minifloat.ts.
inline uint32_t bitsOf(const Format& f, double v) {
    const uint32_t expAllOnes = (1u << f.e) - 1;
    if (std::isnan(v)) return (expAllOnes << f.m) | (1u << (f.m - 1));
    const uint32_t sign = std::signbit(v) ? 1u << (f.e + f.m) : 0u;
    const double a = std::fabs(v);
    if (std::isinf(a)) return sign | (expAllOnes << f.m);
    if (a == 0) return sign;
    const int k = std::ilogb(a);
    if (k < emin(f)) return sign | (uint32_t)(a / std::ldexp(1.0, emin(f) - f.m));       // subnormal
    const uint32_t M = (uint32_t)((a / std::ldexp(1.0, k) - 1.0) * std::ldexp(1.0, f.m));  // exact
    return sign | ((uint32_t)(k + f.bias) << f.m) | M;
}

// The bits the format stores for x.
inline uint32_t encode(const Format& f, double x, Mode mode = Mode::Nearest) { return bitsOf(f, round(f, x, mode)); }

// The value of a bit pattern of the format.
inline double decode(const Format& f, uint32_t bits) {
    const uint32_t expAllOnes = (1u << f.e) - 1;
    const bool neg = (bits >> (f.e + f.m)) & 1u;
    const uint32_t E = (bits >> f.m) & expAllOnes;
    const uint32_t M = bits & ((1u << f.m) - 1);
    double v;
    if (E == expAllOnes) v = M ? std::numeric_limits<double>::quiet_NaN() : std::numeric_limits<double>::infinity();
    else if (E == 0) v = std::ldexp((double)M, emin(f) - f.m);
    else v = std::ldexp(1.0 + std::ldexp((double)M, -f.m), (int)E - f.bias);
    return neg ? -v : v;
}

}  // namespace fpformat
