// pmpp CPU emulator: half precision, the subset of <cuda_fp16.h> the course uses.
// Reached through the stand-in header pmpp/emu/shim/cuda_fp16.h, so a .cu file must
// #include <cuda_fp16.h> to use it, as with nvcc. Needs pmpp/emu/cuda_emu.h first (the
// build force-includes it).
//
// Names and signatures: CUDA Math API Reference Manual 13.5, read 2026-11-21
// (notes/today/infra-facts.md E8): "Half Precision Conversion and Data Movement"
// (group__CUDA__MATH____HALF__MISC), "Half Arithmetic Functions" (____HALF__ARITHMETIC),
// "Half2 Arithmetic Functions" (____HALF2__ARITHMETIC), and the struct pages for __half and
// __half2. Every function the manual marks __host__ __device__ is callable from both here, as
// everything is.
//
// MODELLED, bit for bit: the IEEE binary16 format (1 sign, 5 exponent, 10 fraction bits,
// subnormals, infinities); round-to-nearest-even for every conversion from float and for
// __hadd/__hsub/__hmul/__hdiv and their half2 forms; __hfma and __hfma2 rounded ONCE, as the
// manual says. How: a binary16 sum, difference or product is exact in binary64, a quotient
// rounded to binary64 first rounds correctly to binary16 (53 >= 2 x 11 + 2 bits), and a
// fused multiply-add uses the exact error of the binary64 sum (TwoSum) to break ties.
// NOT MODELLED: NaN payloads (every NaN result is 0x7FFF; a GPU's canonical NaN was not
// checked), the _sat, _relu, _rn and directed-rounding variants, comparisons other than
// through the conversion to float, __shfl_*_sync and atomics on __half and __half2,
// __nv_bfloat16 (<cuda_bf16.h>), and anything about speed. The struct members are a subset:
// constructors from float, double, the integer types and __half_raw, and one conversion operator, to float (the manual lists more). Code compiled
// for the emulator is host code, so contraction and flush-to-zero follow the host compiler.
#pragma once
#include <cmath>
#include <cstdint>
#include <cstring>

namespace pmpp_emu {
namespace fp16 {
// The binary16 value nearest to x + t, where t is a correction far smaller than x's binary64
// ulp whose sign alone matters (-1, 0 or +1): ties go to the side t points to, or to even.
inline uint16_t round(double x, int t = 0) {
  if (std::isnan(x)) return 0x7FFF;
  const uint16_t sign = std::signbit(x) ? 0x8000 : 0;
  double a = std::fabs(x);
  if (sign) t = -t;
  if (std::isinf(a)) return sign | 0x7C00;
  if (a == 0.0) return sign;
  int e;
  std::frexp(a, &e);                                   // a = m 2^e, 0.5 <= m < 1
  const int E = e - 1 < -14 ? -14 : e - 1;             // the binade, or the subnormal range
  const double ulp = std::ldexp(1.0, E - 10);
  const double q = a / ulp, fl = std::floor(q), frac = q - fl;   // exact: ulp is a power of two
  const bool up = frac > 0.5 || (frac == 0.5 && (t > 0 || (t == 0 && std::fmod(fl, 2.0) != 0.0)));
  const double r = (fl + (up ? 1.0 : 0.0)) * ulp;
  if (r >= 65536.0) return sign | 0x7C00;              // 65520 and above round to infinity
  if (r < std::ldexp(1.0, -14)) return sign | uint16_t(r / std::ldexp(1.0, -24));
  int e2;
  std::frexp(r, &e2);
  const int E2 = e2 - 1;
  return sign | uint16_t((E2 + 15) << 10) | uint16_t(uint16_t(r / std::ldexp(1.0, E2 - 10)) - 1024);
}
inline double value(uint16_t h) {
  const int E = (h >> 10) & 31, m = h & 1023;
  double v;
  if (E == 0) v = std::ldexp(double(m), -24);
  else if (E == 31) v = m ? std::nan("") : INFINITY;
  else v = std::ldexp(double(m + 1024), E - 25);
  return (h & 0x8000) ? -v : v;
}
// a * b + c rounded once to binary16.
inline uint16_t fma(uint16_t a, uint16_t b, uint16_t c) {
  const double p = value(a) * value(b), cc = value(c);   // the product is exact
  const double s = p + cc;
  if (!std::isfinite(s)) return round(s);
  const double bb = s - p, err = (p - (s - bb)) + (cc - bb);   // TwoSum: p + cc == s + err exactly
  return round(s, err > 0 ? 1 : (err < 0 ? -1 : 0));
}
}  // namespace fp16
}  // namespace pmpp_emu

struct __half_raw { unsigned short x; };
struct __half2_raw { unsigned short x, y; };

struct __half {
  unsigned short pmpp_bits;     // the binary16 bit pattern
  __half() = default;           // uninitialised, as documented
  constexpr __half(const __half_raw& hr) : pmpp_bits(hr.x) {}
  __half(float f) : pmpp_bits(::pmpp_emu::fp16::round(f)) {}
  __half(double d) : pmpp_bits(::pmpp_emu::fp16::round(d)) {}   // rounded once (binary64 to binary16)
  __half(int i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(unsigned int i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(short i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(unsigned short i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(long long i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(unsigned long long i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(long i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half(unsigned long i) : pmpp_bits(::pmpp_emu::fp16::round(double(i))) {}
  __half& operator=(const __half_raw& hr) { pmpp_bits = hr.x; return *this; }
  __half& operator=(float f) { pmpp_bits = ::pmpp_emu::fp16::round(f); return *this; }
  operator float() const { return float(::pmpp_emu::fp16::value(pmpp_bits)); }
  operator __half_raw() const { return __half_raw{pmpp_bits}; }
};
// 4-byte alignment for __half2 is from memory of cuda_fp16.h, not from the manual.
struct alignas(4) __half2 {
  __half x, y;                  // x is the low half, y the high half
  __half2() = default;
  constexpr __half2(const __half& a, const __half& b) : x(a), y(b) {}
  __half2(const __half2_raw& r) : x(__half_raw{r.x}), y(__half_raw{r.y}) {}
  operator __half2_raw() const { return __half2_raw{x.pmpp_bits, y.pmpp_bits}; }
};
static_assert(sizeof(__half) == 2 && sizeof(__half2) == 4, "binary16 storage");

namespace pmpp_emu {
namespace fp16 {
inline __half mk(uint16_t b) { return __half(__half_raw{b}); }
inline uint16_t bits(__half h) { return h.pmpp_bits; }
}  // namespace fp16
}  // namespace pmpp_emu

// ---- conversion and data movement ---------------------------------------------------------------
inline __half __float2half(const float a) { return ::pmpp_emu::fp16::mk(::pmpp_emu::fp16::round(a)); }
inline __half __float2half_rn(const float a) { return __float2half(a); }
inline float __half2float(const __half a) { return float(::pmpp_emu::fp16::value(a.pmpp_bits)); }
inline __half2 __float2half2_rn(const float a) { return __half2(__float2half(a), __float2half(a)); }
inline __half2 __floats2half2_rn(const float a, const float b) { return __half2(__float2half(a), __float2half(b)); }
inline __half2 __float22half2_rn(const float2 a) { return __floats2half2_rn(a.x, a.y); }
inline float2 __half22float2(const __half2 a) { return float2{__half2float(a.x), __half2float(a.y)}; }
inline __half2 __halves2half2(const __half a, const __half b) { return __half2(a, b); }
inline __half2 make_half2(const __half x, const __half y) { return __half2(x, y); }
inline __half2 __half2half2(const __half a) { return __half2(a, a); }
inline float __low2float(const __half2 a) { return __half2float(a.x); }
inline float __high2float(const __half2 a) { return __half2float(a.y); }
inline __half __low2half(const __half2 a) { return a.x; }
inline __half __high2half(const __half2 a) { return a.y; }
inline unsigned short int __half_as_ushort(const __half h) { return h.pmpp_bits; }
inline __half __ushort_as_half(const unsigned short int i) { return ::pmpp_emu::fp16::mk(i); }
inline short int __half_as_short(const __half h) { return short(h.pmpp_bits); }
inline __half __short_as_half(const short int i) { return ::pmpp_emu::fp16::mk(uint16_t(i)); }
inline __half __int2half_rn(const int i) { return ::pmpp_emu::fp16::mk(::pmpp_emu::fp16::round(double(i))); }
inline int __half2int_rn(const __half h) { return int(std::nearbyint(::pmpp_emu::fp16::value(h.pmpp_bits))); }

// ---- arithmetic ---------------------------------------------------------------------------------
#define PMPP_FP16_BIN(NAME, OP)                                                                          \
  inline __half NAME(const __half a, const __half b) {                                                   \
    return ::pmpp_emu::fp16::mk(::pmpp_emu::fp16::round(::pmpp_emu::fp16::value(a.pmpp_bits) OP         \
                                                        ::pmpp_emu::fp16::value(b.pmpp_bits)));          \
  }
PMPP_FP16_BIN(__hadd, +)
PMPP_FP16_BIN(__hsub, -)
PMPP_FP16_BIN(__hmul, *)
PMPP_FP16_BIN(__hdiv, /)
#undef PMPP_FP16_BIN
inline __half __hfma(const __half a, const __half b, const __half c) {
  return ::pmpp_emu::fp16::mk(::pmpp_emu::fp16::fma(a.pmpp_bits, b.pmpp_bits, c.pmpp_bits));
}
inline __half __hneg(const __half a) { return ::pmpp_emu::fp16::mk(uint16_t(a.pmpp_bits ^ 0x8000)); }
inline __half __habs(const __half a) { return ::pmpp_emu::fp16::mk(uint16_t(a.pmpp_bits & 0x7FFF)); }

inline __half2 __hadd2(const __half2 a, const __half2 b) { return __half2(__hadd(a.x, b.x), __hadd(a.y, b.y)); }
inline __half2 __hsub2(const __half2 a, const __half2 b) { return __half2(__hsub(a.x, b.x), __hsub(a.y, b.y)); }
inline __half2 __hmul2(const __half2 a, const __half2 b) { return __half2(__hmul(a.x, b.x), __hmul(a.y, b.y)); }
inline __half2 __h2div(const __half2 a, const __half2 b) { return __half2(__hdiv(a.x, b.x), __hdiv(a.y, b.y)); }
inline __half2 __hfma2(const __half2 a, const __half2 b, const __half2 c) {
  return __half2(__hfma(a.x, b.x, c.x), __hfma(a.y, b.y, c.y));
}
inline __half2 __hneg2(const __half2 a) { return __half2(__hneg(a.x), __hneg(a.y)); }
inline __half2 __habs2(const __half2 a) { return __half2(__habs(a.x), __habs(a.y)); }

// ---- operators (the manual: "see __hadd", "see __hadd2", ...) --------------------------------------
inline __half operator+(const __half& lh, const __half& rh) { return __hadd(lh, rh); }
inline __half operator-(const __half& lh, const __half& rh) { return __hsub(lh, rh); }
inline __half operator*(const __half& lh, const __half& rh) { return __hmul(lh, rh); }
inline __half operator/(const __half& lh, const __half& rh) { return __hdiv(lh, rh); }
inline __half& operator+=(__half& lh, const __half& rh) { return lh = __hadd(lh, rh); }
inline __half operator-(const __half& h) { return __hneg(h); }
inline __half2 operator+(const __half2& h) { return h; }
inline __half2 operator-(const __half2& h) { return __hneg2(h); }
inline __half2 operator+(const __half2& lh, const __half2& rh) { return __hadd2(lh, rh); }
inline __half2 operator-(const __half2& lh, const __half2& rh) { return __hsub2(lh, rh); }
inline __half2 operator*(const __half2& lh, const __half2& rh) { return __hmul2(lh, rh); }
inline __half2 operator/(const __half2& lh, const __half2& rh) { return __h2div(lh, rh); }
inline __half2& operator+=(__half2& lh, const __half2& rh) { return lh = __hadd2(lh, rh); }
inline __half2& operator-=(__half2& lh, const __half2& rh) { return lh = __hsub2(lh, rh); }
inline __half2& operator*=(__half2& lh, const __half2& rh) { return lh = __hmul2(lh, rh); }
inline __half2& operator/=(__half2& lh, const __half2& rh) { return lh = __h2div(lh, rh); }
