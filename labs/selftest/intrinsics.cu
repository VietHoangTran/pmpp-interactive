// Self-test: 8- and 16-bit vector types, integer and type-casting intrinsics, single-precision
// math. Expected values follow the CUDA Math API 13.5 and Programming Guide 13.5 as recorded in
// notes/today/infra-facts.md, group E. On CUDA, __dp4a needs compute capability 6.1 or higher
// (x2-facts A2). HIP has no __dp4a (x2-facts B1); this file is not built for HIP.
#include <climits>
#include <cmath>
#include "selftest_cu.h"

// ---- sizes and alignments: CUDA Programming Guide 13.5, Table 49 -------------------------------
#define ST_VEC(T, S, A) static_assert(sizeof(T) == S && alignof(T) == A, #T ": size or alignment differs from Table 49");
ST_VEC(char1, 1, 1) ST_VEC(char2, 2, 2) ST_VEC(char3, 3, 1) ST_VEC(char4, 4, 4)
ST_VEC(uchar1, 1, 1) ST_VEC(uchar2, 2, 2) ST_VEC(uchar3, 3, 1) ST_VEC(uchar4, 4, 4)
ST_VEC(short1, 2, 2) ST_VEC(short2, 4, 4) ST_VEC(short3, 6, 2) ST_VEC(short4, 8, 8)
ST_VEC(ushort1, 2, 2) ST_VEC(ushort2, 4, 4) ST_VEC(ushort3, 6, 2) ST_VEC(ushort4, 8, 8)
ST_VEC(int2, 8, 8) ST_VEC(int3, 12, 4) ST_VEC(int4, 16, 16) ST_VEC(uint4, 16, 16)
ST_VEC(float2, 8, 8) ST_VEC(float3, 12, 4) ST_VEC(float4, 16, 16)
ST_VEC(double2, 16, 16) ST_VEC(double3, 24, 8)
#undef ST_VEC

// ---- __dp4a, __byte_perm, high halves of products, bit casts -----------------------------------
__global__ void kIntrinsics(const int* in, int* out) {
    // in: 0x01020304, 0x05060708, 0xFF80017F (bytes 127, 1, -128, -1), 0x7F7F7F7F
    const char4 c = make_char4(127, 1, -128, -1), c127 = make_char4(127, 127, 127, 127);
    const uchar4 u = make_uchar4(127, 1, 128, 255), u127 = make_uchar4(127, 127, 127, 127);
    int k = 0;
    out[k++] = __dp4a(in[0], in[1], 10);                                   // 5 + 12 + 21 + 32 + 10
    out[k++] = __dp4a(in[2], in[3], 0);                                    // signed bytes: -127
    out[k++] = (int)__dp4a((unsigned)in[2], (unsigned)in[3], 0u);          // unsigned bytes: 127 * 511
    out[k++] = __dp4a(c, c127, 0);                                         // the char4 form: -127
    out[k++] = (int)__dp4a(u, u127, 1u);                                   // the uchar4 form: 64898
    out[k++] = __dp4a(in[3], in[3], INT_MAX);                              // 32-bit wrap (PTX dp4a.s32)
    out[k++] = (int)__byte_perm(0x03020100u, 0x07060504u, 0x7531u);       // bytes 1, 3, 5, 7
    out[k++] = (int)__byte_perm(0x03020111u, 0x07060504u, 0xFFFF8888u);   // bit 3 and bits 16..31 unused
    out[k++] = (int)__byte_perm(0x03020100u, 0x07060504u, 0x0123u);       // bytes 3, 2, 1, 0
    out[k++] = (int)__umulhi(0xFFFFFFFFu, 0xFFFFFFFFu);                    // 0xFFFFFFFE
    out[k++] = __mulhi(-1, 1);                                             // -1
    out[k++] = __mulhi(0x40000000, 8);                                     // 2
    out[k++] = (int)__umul64hi(0x8000000000000000ull, 4ull);                // 2
    out[k++] = (int)__float_as_uint(1.0f);                                 // 0x3F800000
    out[k++] = __float_as_int(-0.0f);                                      // INT_MIN
    out[k++] = __float_as_int(__uint_as_float(0x40490FDBu) * 2.0f);        // 2 pi: 0x40C90FDB
    out[k++] = __float_as_int(__int_as_float(0x7F800000));                 // +infinity survives
}
static std::string tIntrinsics() {
    StBuf<int> in(std::vector<int>{0x01020304, 0x05060708, (int)0xFF80017Fu, 0x7F7F7F7F}), out(17, 0);
    kIntrinsics<<<1, 1>>>(in.dev, out.dev);
    ST_LAUNCHED();
    const std::vector<int> o = out.host();
    const long long wrapped = (long long)INT_MAX + 4 * 127 * 127 - (1ll << 32);
    const int want[17] = {80, -127, 127 * 511, -127, 127 * 511 + 1, (int)wrapped, 0x07050301, 0x11111111, 0x00010203,
                          (int)0xFFFFFFFEu, -1, 2, 2, 0x3F800000, INT_MIN, 0x40C90FDB, 0x7F800000};
    const char* what[17] = {"__dp4a(int)", "__dp4a(int), negative bytes", "__dp4a(unsigned)", "__dp4a(char4)",
                            "__dp4a(uchar4)", "__dp4a wraps at 32 bits", "__byte_perm 0x7531", "__byte_perm unused bits",
                            "__byte_perm 0x0123", "__umulhi", "__mulhi(-1, 1)", "__mulhi", "__umul64hi",
                            "__float_as_uint", "__float_as_int(-0)", "__uint_as_float", "__int_as_float"};
    for (int i = 0; i < 17; ++i) ST_REQUIRE(o[i] == want[i], "%s gave 0x%08x, expected 0x%08x", what[i], (unsigned)o[i], (unsigned)want[i]);
    return "";
}

// ---- single-precision math: values, not the GPU's error bounds ---------------------------------
__global__ void kMath(const float* in, float* out) {
    const float one = in[0], two = in[1], three = in[2], half = in[3], big = in[4];
    int k = 0;
    out[k++] = expf(one);          out[k++] = __expf(one);
    out[k++] = exp2f(three);       out[k++] = log2f(two * two * two);
    out[k++] = logf(one);          out[k++] = __logf(two);
    out[k++] = rsqrtf(two * two);  out[k++] = sqrtf(two * two);
    out[k++] = fmaf(two, three, one);
    out[k++] = tanhf(half);        out[k++] = erff(half);
    out[k++] = fminf(two, -three); out[k++] = fmaxf(two, -three); out[k++] = fabsf(-three);
    out[k++] = __fdividef(one, three);
    out[k++] = __powf(two, three); out[k++] = powf(two, half);
    out[k++] = __saturatef(big);   out[k++] = __saturatef(-half); out[k++] = __saturatef(half);
    out[k++] = __sinf(half);       out[k++] = __cosf(half);     out[k++] = copysignf(three, -one);
}
static std::string tMath() {
    StBuf<float> in(std::vector<float>{1.f, 2.f, 3.f, 0.5f, 7.f}), out(23, -1.f);
    kMath<<<1, 1>>>(in.dev, out.dev);
    ST_LAUNCHED();
    const std::vector<float> o = out.host();
    const double e = std::exp(1.0);
    const double want[23] = {e, e, 8, 3, 0, std::log(2.0), 0.5, 2, 7, std::tanh(0.5), std::erf(0.5), -3, 2, 3,
                             1.0 / 3, 8, std::sqrt(2.0), 1, 0, 0.5, std::sin(0.5), std::cos(0.5), -3};
    const char* what[23] = {"expf", "__expf", "exp2f", "log2f", "logf", "__logf", "rsqrtf", "sqrtf", "fmaf", "tanhf", "erff",
                            "fminf", "fmaxf", "fabsf", "__fdividef", "__powf", "powf", "__saturatef(7)", "__saturatef(-0.5)",
                            "__saturatef(0.5)", "__sinf", "__cosf", "copysignf"};
    for (int i = 0; i < 23; ++i) {
        // A loose tolerance: the fast intrinsics are approximations on a GPU, and NVIDIA's error
        // bounds are not what this test checks.
        ST_REQUIRE(std::fabs(o[i] - want[i]) <= 1e-5 * std::fmax(1.0, std::fabs(want[i])), "%s gave %.9g, expected %.9g",
                   what[i], o[i], want[i]);
    }
    return "";
}

extern const pmpp_selftest_case pmpp_selftest_cases[] = {
    {"__dp4a (int, unsigned, char4, uchar4), __byte_perm, __umulhi, __mulhi, __umul64hi, bit casts", tIntrinsics},
    {"single-precision math: expf, exp2f, log2f, rsqrtf, fmaf, tanhf, erff, fminf, fmaxf, fabsf, __expf, __fdividef, ...", tMath},
};
ST_END_CASES
