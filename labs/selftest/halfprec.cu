// Self-test: __half and __half2 from <cuda_fp16.h>, the subset the emulator provides
// (pmpp/emu/cuda_fp16_emu.h). Every expected value is a binary16 bit pattern worked out by
// hand from IEEE binary16 and round-to-nearest-even, which the CUDA Math API 13.5 states for
// these functions (notes/today/infra-facts.md E8). On CUDA, half arithmetic in device code
// needs compute capability 5.3 or higher (from memory of cuda_fp16.h; not checked).
// This file is not built for HIP, whose header is <hip/hip_fp16.h>.
#include <cuda_fp16.h>
#include "selftest_cu.h"

// ---- every bit pattern: half -> float -> half is the identity (NaNs stay NaNs) ------------------
__global__ void kRoundTrip(unsigned* bad) {
    unsigned wrong = 0;
    for (unsigned i = threadIdx.x; i < 65536u; i += blockDim.x) {
        const __half h = __ushort_as_half((unsigned short)i);
        const unsigned short back = __half_as_ushort(__float2half(__half2float(h)));
        const bool nan = (i & 0x7C00u) == 0x7C00u && (i & 0x3FFu) != 0;
        wrong += nan ? !((back & 0x7C00u) == 0x7C00u && (back & 0x3FFu) != 0) : back != i;
    }
    atomicAdd(bad, wrong);
}
static std::string tRoundTrip() {
    StBuf<unsigned> bad(1, 0u);
    kRoundTrip<<<1, 256>>>(bad.dev);
    ST_LAUNCHED();
    ST_REQUIRE(bad.host()[0] == 0, "%u of 65536 bit patterns did not survive half -> float -> half", bad.host()[0]);
    return "";
}

// ---- conversions from float: round to nearest, ties to even; subnormals; overflow -------------
__global__ void kFromFloat(const float* in, unsigned short* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = __half_as_ushort(__float2half_rn(in[i]));
}
static std::string tFromFloat() {
    const float t = 1.0f / 2048, s = 1.0f / 16777216;                 // 2^-11 and 2^-24
    const std::vector<float> in = {1.0f, -2.5f, 65504.0f, 65519.0f, 65520.0f, 1e9f, s, s / 2, 3 * s / 2, s / 4,
                                   1 + t, 1 + 3 * t, 1.0f / 3, -1e-8f, 6.103515625e-05f, 0.0f};
    const unsigned short want[] = {0x3C00, 0xC100, 0x7BFF, 0x7BFF, 0x7C00, 0x7C00, 0x0001, 0x0000, 0x0002, 0x0000,
                                   0x3C00, 0x3C02, 0x3555, 0x8000, 0x0400, 0x0000};
    const int n = (int)in.size();
    StBuf<float> a(in);
    StBuf<unsigned short> out(n, 0xFFFF);
    kFromFloat<<<1, 1>>>(a.dev, out.dev, n);
    ST_LAUNCHED();
    const std::vector<unsigned short> o = out.host();
    for (int i = 0; i < n; ++i)
        ST_REQUIRE(o[i] == want[i], "__float2half_rn(%.9g) gave 0x%04x, expected 0x%04x", in[i], o[i], want[i]);
    return "";
}

// ---- arithmetic: one rounding each; __hfma rounds once where mul-then-add rounds twice ---------
__global__ void kArith(unsigned short* out, float* f) {
    const __half a = __ushort_as_half(0x3C01), c = __ushort_as_half(0x1000);   // 1 + 2^-10 and 2^-11
    const __half one = __float2half(1.0f), three = __float2half(3.0f), big = __ushort_as_half(0x7BFF);
    int k = 0;
    out[k++] = __half_as_ushort(__hfma(a, a, c));                      // exact 1 + 2^-9 + 2^-11 + 2^-20: up
    out[k++] = __half_as_ushort(__hadd(__hmul(a, a), c));              // 1 + 2^-9, then a tie: to even
    out[k++] = __half_as_ushort(__hdiv(one, three));                   // 1/3
    out[k++] = __half_as_ushort(__hadd(big, __float2half(16.0f)));     // 65520: a tie that rounds to infinity
    out[k++] = __half_as_ushort(__hadd(big, __float2half(8.0f)));      // 65512: back to 65504
    out[k++] = __half_as_ushort(__hsub(one, three));                   // -2
    out[k++] = __half_as_ushort(__hneg(three));
    out[k++] = __half_as_ushort(__habs(__hneg(three)));
    out[k++] = __half_as_ushort(one + three * three - one / three);    // operators: 1 + 9 - 0.33325 = 9.66675 -> 9.6640625
    __half acc = one;
    acc += three;
    out[k++] = __half_as_ushort(-acc);                                 // -4
    out[k++] = __half_as_ushort(__int2half_rn(2049));                  // a tie between 2048 and 2050: 2048
    out[k++] = (unsigned short)__half2int_rn(__float2half(2.5f));      // 2
    out[k++] = (unsigned short)__half2int_rn(__float2half(3.5f));      // 4
    // half2: lane by lane, low half first
    const __half2 v = __floats2half2_rn(1.5f, -2.0f), w = __halves2half2(a, three);
    const __half2 fm = __hfma2(__halves2half2(a, three), w, __halves2half2(c, one));   // (fma(a, a, c), 3 * 3 + 1)
    out[k++] = __half_as_ushort(__low2half(fm));
    out[k++] = __half_as_ushort(__high2half(fm));
    const __half2 sum = __hadd2(v, __float2half2_rn(0.25f)), prod = v * w, quot = __h2div(v, __half2half2(three));
    out[k++] = __half_as_ushort(__low2half(sum));
    out[k++] = __half_as_ushort(__high2half(sum));
    out[k++] = __half_as_ushort(__high2half(prod));                    // -2 * 3
    out[k++] = __half_as_ushort(__low2half(quot));                     // 1.5 / 3 = 0.5
    out[k++] = __half_as_ushort(__high2half(__hneg2(v)));              // 2
    out[k++] = __half_as_ushort(__low2half(__habs2(__hsub2(w, __float2half2_rn(4.0f)))));   // |1 + 2^-10 - 4| = 3 - 2^-10, a tie: to even, 3
    f[0] = __low2float(v);
    f[1] = __high2float(v);
    const float2 g = __half22float2(__float22half2_rn(make_float2(0.1f, 1e5f)));
    f[2] = g.x;
    f[3] = g.y;
    f[4] = (float)three;                                               // the conversion operator
}
static std::string tArith() {
    StBuf<unsigned short> out(21, 0xFFFF);
    StBuf<float> f(5, -1.f);
    kArith<<<1, 1>>>(out.dev, f.dev);
    ST_LAUNCHED();
    const std::vector<unsigned short> o = out.host();
    const unsigned short want[21] = {0x3C03, 0x3C02, 0x3555, 0x7C00, 0x7BFF, 0xC000, 0xC200, 0x4200, 0x48D5, 0xC400,
                                     0x6800, 2, 4, 0x3C03, 0x4900, 0x3F00, 0xBF00, 0xC600, 0x3800, 0x4000, 0x4200};
    const char* what[21] = {"__hfma (one rounding)", "__hmul then __hadd (two roundings)", "__hdiv(1, 3)",
                            "65504 + 16", "65504 + 8", "__hsub", "__hneg", "__habs", "operators + * - /",
                            "operator+= and unary -", "__int2half_rn(2049)", "__half2int_rn(2.5)", "__half2int_rn(3.5)",
                            "__hfma2 low", "__hfma2 high", "__hadd2 low", "__hadd2 high", "operator* on half2",
                            "__h2div", "__hneg2", "__habs2 and __hsub2"};
    for (int i = 0; i < 21; ++i) ST_REQUIRE(o[i] == want[i], "%s gave 0x%04x, expected 0x%04x", what[i], o[i], want[i]);
    const std::vector<float> g = f.host();
    ST_REQUIRE(g[0] == 1.5f && g[1] == -2.0f, "__low2float / __high2float gave %g, %g", g[0], g[1]);
    ST_REQUIRE(g[2] == 0.0999755859375f && g[3] == INFINITY, "__float22half2_rn / __half22float2 gave %.9g, %.9g", g[2], g[3]);
    ST_REQUIRE(g[4] == 3.0f, "(float)__half gave %g", g[4]);
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"__half: every bit pattern survives half -> float -> half", tRoundTrip},
    {"__float2half_rn: ties to even, subnormals, overflow to infinity", tFromFloat},
    {"__half / __half2 arithmetic: one rounding each, __hfma and __hfma2 fused", tArith},
};
ST_END_CASES
