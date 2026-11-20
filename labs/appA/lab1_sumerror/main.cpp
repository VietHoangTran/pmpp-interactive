// Lab A.1 driver: three float sums in three accumulator types, judged against a CPU reference
// and a CPU model of the same order. You do not need to edit this file; your work goes in work/.
//
// In order:
//   1. data sets          deterministic, the same on every platform
//   2. the CPU reference  a Neumaier-compensated sum in double
//   3. the CPU model      the kernels' exact orders, every addition rounded in software
//   4. error bounds       README.md ("How the tests judge a result") derives them
//   5. device glue        your launchers (CUDA, HIP, emulators) or encoders (Metal), the probe
//   6. tests
//   7. benchmark          the rows sections A.4 and A.6 and the labs page read
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <string>
#include <vector>

#include "pmpp/harness.hpp"
#include "fpformat.h"
#include "sumerror.h"

using fpformat::Format;
using fpformat::Mode;
using pmpp::Buf;

enum Method { MSerial = 0, MTree = 1, MKahan = 2 };
static const char* methodName(Method m) { return m == MSerial ? "serial" : m == MTree ? "tree" : "kahan"; }
static const char* accName(Acc a) { return a == AccHalf ? "half" : a == AccBfloat ? "bfloat" : "float"; }
static const Format& formatOf(Acc a) {
    return a == AccHalf ? fpformat::FP16 : a == AccBfloat ? fpformat::BF16 : fpformat::FP32;
}

// ======================================================================= 1. data
enum class Set { Ones, Ints, Uniform, Mixed, MixedAsc, MixedDesc, Cancel, Softmax, BigSmall };
static const char* setName(Set s) {
    switch (s) {
    case Set::Ones: return "ones";
    case Set::Ints: return "ints";
    case Set::Uniform: return "uniform";
    case Set::Mixed: return "mixed";
    case Set::MixedAsc: return "mixed-asc";
    case Set::MixedDesc: return "mixed-desc";
    case Set::Cancel: return "cancel";
    case Set::Softmax: return "softmax";
    case Set::BigSmall: return "big-small";
    }
    return "?";
}

struct Rng {   // xorshift32, as pmpp::random_floats
    uint32_t x;
    explicit Rng(uint32_t seed) : x(seed * 2654435761u + 1u) {}
    uint32_t next() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
    double unit() { return (double)(next() >> 8) / 16777216.0; }   // [0, 1), 24 bits
};

// n values of set s, rounded to the accumulator's format (so converting them to the
// accumulator is exact), with any value below the format's smallest normal number set to 0
// (no subnormal inputs; Metal may flush them, appA-facts C2). Stored as float32.
//   ones       1
//   ints       whole numbers from -3 to 3
//   uniform    uniform on [0, 1)
//   mixed      magnitudes from 2^-14 to 2^-2: each binade equally likely, uniform inside it
//   mixed-asc, mixed-desc   the same values sorted ascending, descending
//   cancel     mixed with random signs
//   softmax    exp(l - max) for logits l uniform on [-9, 0], the largest (exp 1) at a random place
//   big-small  1, then n-1 copies of half an ULP of 1 in the format (2^-24 for float)
static std::vector<float> makeData(Set s, int n, Acc acc) {
    const Format& f = formatOf(acc);
    Rng r(1000u + 7919u * (uint32_t)s + (uint32_t)n);
    std::vector<double> v((size_t)n);
    for (int i = 0; i < n; ++i) {
        switch (s) {
        case Set::Ones: v[i] = 1.0; break;
        case Set::Ints: v[i] = (double)(int)(r.next() % 7u) - 3.0; break;
        case Set::Uniform: v[i] = r.unit(); break;
        case Set::Mixed: case Set::MixedAsc: case Set::MixedDesc: case Set::Cancel: {
            const int e = -14 + (int)(r.next() % 12u);
            v[i] = std::ldexp(1.0 + (double)(r.next() >> 9) / 8388608.0, e);
            if (s == Set::Cancel && (r.next() & 1u)) v[i] = -v[i];
            break;
        }
        case Set::Softmax: v[i] = std::exp(-9.0 * r.unit()); break;
        case Set::BigSmall: v[i] = i == 0 ? 1.0 : std::ldexp(1.0, -(f.m + 1)); break;
        }
    }
    if (s == Set::Softmax && n > 0) v[r.next() % (uint32_t)n] = 1.0;
    std::vector<float> x((size_t)n);
    for (int i = 0; i < n; ++i) {
        double q = fpformat::round(f, v[i]);
        if (std::fabs(q) < fpformat::minNormal(f)) q = 0.0;
        x[i] = (float)q;
    }
    if (s == Set::MixedAsc) std::sort(x.begin(), x.end());
    if (s == Set::MixedDesc) std::sort(x.begin(), x.end(), [](float a, float b) { return a > b; });
    return x;
}

// ======================================================================= 2. reference
// Neumaier's variant of Kahan's compensated sum, in double: far more accurate than any sum
// the device computes in float or narrower (appA-facts E1). sumAbs is the sum of |x|, the
// scale of every error bound.
struct Ref { double sum, sumAbs; };
static Ref reference(const std::vector<float>& x) {
    double s = 0, c = 0, a = 0;
    for (float f : x) {
        const double v = f;
        const double t = s + v;
        if (std::fabs(s) >= std::fabs(v)) c += (s - t) + v;
        else c += (v - t) + s;
        s = t;
        a += std::fabs(v);
    }
    return {s + c, a};
}

// ======================================================================= 3. the CPU model
// The same additions in the same order as the kernels, each rounded to the format in
// software (fpformat.h). A kernel whose result has the same bits as this has added in the
// order the lab specifies, with one rounding to the accumulator per addition.
static double modelSerial(const std::vector<float>& x, const Format& f, Mode md) {
    double s = 0;
    for (float v : x) s = fpformat::add(f, s, v, md);
    return s;
}

// One pass of sum_tree / sum_kahan: one sum per block.
static std::vector<double> modelPass(const std::vector<double>& in, int coarse, int block, bool kahan,
                                     const Format& f, Mode md) {
    const size_t n = in.size();
    const size_t per = (size_t)block * coarse;
    const size_t blocks = (n + per - 1) / per;
    std::vector<double> out(blocks), s((size_t)block);
    for (size_t b = 0; b < blocks; ++b) {
        for (int t = 0; t < block; ++t) {
            double sum = 0, c = 0;
            for (int j = 0; j < coarse; ++j) {
                const size_t i = b * per + (size_t)t + (size_t)j * block;
                if (i >= n) break;
                if (!kahan) {
                    sum = fpformat::add(f, sum, in[i], md);
                } else {
                    const double y = fpformat::sub(f, in[i], c, md);
                    const double nx = fpformat::add(f, sum, y, md);
                    c = fpformat::sub(f, fpformat::sub(f, nx, sum, md), y, md);
                    sum = nx;
                }
            }
            s[t] = sum;
        }
        for (int stride = block / 2; stride >= 1; stride /= 2)
            for (int t = 0; t < stride; ++t) s[t] = fpformat::add(f, s[t], s[t + stride], md);
        out[b] = s[0];
    }
    return out;
}

static double modelTwoPasses(const std::vector<float>& x, int coarse, int block, bool kahan, const Format& f, Mode md) {
    const std::vector<double> in(x.begin(), x.end());
    const std::vector<double> partials = modelPass(in, coarse, block, kahan, f, md);
    const int coarse2 = (int)((partials.size() + block - 1) / block);
    return modelPass(partials, coarse2, block, kahan, f, md)[0];
}

static double model(Method m, const std::vector<float>& x, int coarse, int block, Acc acc, Mode md) {
    if (m == MSerial) return modelSerial(x, formatOf(acc), md);
    return modelTwoPasses(x, coarse, block, m == MKahan, formatOf(acc), md);
}

// ======================================================================= 4. error bounds
// README.md derives these. u is the format's unit roundoff (2^-24 for float); it is doubled
// because Metal allows round-toward-zero (appA-facts C2), whose error per addition is below 2u.
static double gammaK(double k, double u2) {
    if (k <= 0) return 0;
    const double ku = k * u2;
    return ku >= 1 ? std::numeric_limits<double>::infinity() : ku / (1 - ku);
}
static int log2int(int b) { int l = 0; while ((1 << l) < b) ++l; return l; }
static int blocksOf(int n, int coarse, int block) { return (n + block * coarse - 1) / (block * coarse); }
static int coarse2Of(int n, int coarse, int block) { return (blocksOf(n, coarse, block) + block - 1) / block; }

// The largest |result - reference| allowed, as a multiple of sum|x|.
static double relBound(Method m, Acc acc, int n, int coarse, int block) {
    const double u2 = 2 * fpformat::unitRoundoff(formatOf(acc));
    if (m == MSerial) return gammaK(n - 1, u2);
    const int L = log2int(block), c2 = coarse2Of(n, coarse, block);
    if (m == MTree) return gammaK((coarse - 1) + L + (c2 - 1) + L, u2);
    // A compensated loop over c elements: 2u + O(c u^2) (Kahan; Goldberg's Theorem 8), with
    // u doubled and a generous constant on the second-order term. One element: exact.
    auto comp = [&](int c) { return c > 1 ? 2 * u2 + 4.0 * c * u2 * u2 : 0.0; };
    const double g = gammaK(L, u2);
    return (1 + comp(coarse)) * (1 + g) * (1 + comp(c2)) * (1 + g) - 1;
}

// Plus one smallest normal value per rounding, in case the device flushes a subnormal result
// to zero (Metal may, C2). Only FP16 data can produce one here: every value is a multiple of
// the data's finest spacing, which is 2^-24 for FP16 and far above the smallest normal for
// float (2^-37) and BF16 (2^-21).
static double absAllowance(Method m, Acc acc, int n) {
    return (m == MKahan ? 4.0 : 2.0) * n * fpformat::minNormal(formatOf(acc));
}

// ======================================================================= 5. device glue
struct Outcome {
    enum Kind { Ran, NotBuilt, Skip, Refused } kind;
    std::string why;
    pmpp::Timing time;
};
static const char* launcherName(Method m) {
    return m == MSerial ? "launchSumSerial" : m == MTree ? "launchSumTree" : "launchSumKahan";
}

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeSumSerial(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> x,
                     id<MTLBuffer> out, uint32_t n);
bool encodeSumTree(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> x,
                   id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block);
bool encodeSumKahan(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> x,
                    id<MTLBuffer> partials, id<MTLBuffer> out, uint32_t n, uint32_t coarse, uint32_t block);
void encodeFpProbe(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> out);

static std::string kernelName(Method m, Acc a) { return std::string("sum_") + methodName(m) + "_" + accName(a); }

// Your pipelines, looked up once each (a missing one is reported once, not per case).
static id<MTLComputePipelineState> sumPipeline(Method m, Acc a, std::string* why) {
    static std::map<std::string, id<MTLComputePipelineState>> found;
    static std::map<std::string, std::string> missing;
    const std::string name = kernelName(m, a);
    auto hit = found.find(name);
    if (hit != found.end()) return hit->second;
    auto miss = missing.find(name);
    if (miss == missing.end()) {
        id<MTLComputePipelineState> p = pmpp::metal::pipeline(PMPP_METAL_SOURCE, name);
        if (p) { found[name] = p; return p; }
        std::string err = pmpp::take_error();
        miss = missing.emplace(name, "no kernel " + name + " in kernel.metal" + (err.empty() ? "" : ": " + err)).first;
        std::printf("  note: %s\n", miss->second.c_str());
    }
    if (why) *why = miss->second;
    return nil;
}

static bool accWanted(Acc, std::string*) { return true; }   // Metal: float, half and bfloat

static Outcome runSum(Method m, Acc a, Buf* X, Buf* P, Buf* O, int n, int coarse, int block) {
    std::string why;
    id<MTLComputePipelineState> p = sumPipeline(m, a, &why);
    if (!p) return {Outcome::NotBuilt, why, {0, -1}};
    if (m != MSerial && (NSUInteger)block > p.maxTotalThreadsPerThreadgroup) {
        return {Outcome::Skip, "block " + std::to_string(block) + " is above the " +
                std::to_string((int)p.maxTotalThreadsPerThreadgroup) + " threads " + kernelName(m, a) + " allows here", {0, -1}};
    }
    bool ok = false;
    pmpp::Timing t = pmpp::run([&](void* vp) {
        auto* pass = (pmpp::metal::Pass*)vp;
        id<MTLBuffer> x = pmpp::metal::buffer(X), part = pmpp::metal::buffer(P), out = pmpp::metal::buffer(O);
        if (m == MSerial) ok = encodeSumSerial(pass->encoder, p, x, out, (uint32_t)n);
        else if (m == MTree) ok = encodeSumTree(pass->encoder, p, x, part, out, (uint32_t)n, (uint32_t)coarse, (uint32_t)block);
        else ok = encodeSumKahan(pass->encoder, p, x, part, out, (uint32_t)n, (uint32_t)coarse, (uint32_t)block);
    });
    if (!ok) {
        const char* enc = m == MSerial ? "encodeSumSerial" : m == MTree ? "encodeSumTree" : "encodeSumKahan";
        return {Outcome::Refused, std::string(enc) + " returned false for a shape the pipeline allows", t};
    }
    return {Outcome::Ran, "", t};
}

// The largest power-of-two block up to `want` that every tree and Kahan pipeline here allows.
static int benchBlock(int want) {
    int b = want;
    for (Acc a : {AccFloat, AccHalf, AccBfloat})
        for (Method m : {MTree, MKahan}) {
            id<MTLComputePipelineState> p = sumPipeline(m, a, nullptr);
            if (p) while (b > 1 && (NSUInteger)b > p.maxTotalThreadsPerThreadgroup) b /= 2;
        }
    return b;
}

static bool runProbe(Buf* in, Buf* out, std::string* why) {
    id<MTLComputePipelineState> p = pmpp::metal::pipeline(std::string(PMPP_METAL_SHARED_DIR) + "/fpprobe.metal", "fp_probe");
    if (!p) { *why = "the provided fpprobe.metal did not build: " + pmpp::take_error(); return false; }
    pmpp::run([&](void* vp) {
        auto* pass = (pmpp::metal::Pass*)vp;
        encodeFpProbe(pass->encoder, p, pmpp::metal::buffer(in), pmpp::metal::buffer(out));
    });
    return true;
}

static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
// CUDA, HIP and the emulators: your launchers in kernel.cu.
static bool accWanted(Acc a, std::string* why) {
    if (a == AccFloat) return true;
    if (pmpp::is_emu()) {
        *why = "the CPU emulators do not model __half or __nv_bfloat16 (README, \"Where the backends differ\")";
        return false;
    }
    if (a == AccBfloat && std::string(pmpp::backend()) == "cuda" && pmpp::device_prop("computeCapability.major", 0) < 8) {
        *why = "__nv_bfloat16 needs compute capability 8.0 or higher (appA-facts A2)";
        return false;
    }
    return true;
}

static Outcome runSum(Method m, Acc a, Buf* X, Buf* P, Buf* O, int n, int coarse, int block) {
    bool ok = false;
    pmpp::Timing t = pmpp::run([&](void*) {
        const float* x = (const float*)pmpp::raw(X);
        float* part = (float*)pmpp::raw(P);
        float* out = (float*)pmpp::raw(O);
        if (m == MSerial) ok = launchSumSerial(x, out, n, a);
        else if (m == MTree) ok = launchSumTree(x, part, out, n, coarse, block, a);
        else ok = launchSumKahan(x, part, out, n, coarse, block, a);
    });
    if (!ok) return {Outcome::NotBuilt, std::string(launcherName(m)) + " returned false for the " + accName(a) + " accumulator", t};
    return {Outcome::Ran, "", t};
}

static int benchBlock(int want) {
    int b = want;
    while (b > 1 && b > (int)pmpp::device_prop("maxThreadsPerBlock", 1024)) b /= 2;
    return b;
}

static bool runProbe(Buf* in, Buf* out, std::string*) {
    pmpp::run([&](void*) { launchFpProbe((const float*)pmpp::raw(in), (float*)pmpp::raw(out)); });
    return true;
}

static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- the probe
struct Probe { bool ok; std::string why, rounding; bool keepsDenormals, fusesMulAdd; float raw[3]; };
static Probe probe() {
    Probe r{false, "", "other", false, false, {0, 0, 0}};
    const float in[6] = {1.0f, 0x1.8p-24f, 0x1p-126f, 0.5f, 1.0f + 0x1p-12f, -(1.0f + 0x1p-11f)};
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float init[3] = {nan, nan, nan};
    Buf* bi = pmpp::alloc(sizeof in);
    Buf* bo = pmpp::alloc(sizeof init);
    pmpp::upload(bi, in, sizeof in);
    pmpp::upload(bo, init, sizeof init);
    if (runProbe(bi, bo, &r.why)) {
        pmpp::download(bo, r.raw, sizeof r.raw);
        const float above1 = std::nextafter(1.0f, 2.0f);
        r.rounding = r.raw[0] == above1 ? "nearest" : r.raw[0] == 1.0f ? "toward-zero" : "other";
        r.keepsDenormals = r.raw[1] == 0x1p-127f;
        r.fusesMulAdd = r.raw[2] == 0x1p-24f;
        const bool known = r.rounding != "other" && (r.keepsDenormals || r.raw[1] == 0.0f) &&
                           (r.fusesMulAdd || r.raw[2] == 0.0f);
        r.ok = known;
        if (!known) {
            char b[160];
            std::snprintf(b, sizeof b, "unexpected probe results %a, %a, %a", r.raw[0], r.raw[1], r.raw[2]);
            r.why = b;
        }
    }
    pmpp::release(bi);
    pmpp::release(bo);
    return r;
}
static Mode modeOf(const Probe& p) { return p.rounding == "toward-zero" ? Mode::TowardZero : Mode::Nearest; }

// ---------------------------------------------------------------- one problem on the device
struct DeviceProblem {
    Buf *X, *P, *O;
    int capN, capPartials;
    DeviceProblem(int maxN, int minBlock) : capN(maxN), capPartials((maxN + minBlock - 1) / minBlock) {
        X = pmpp::alloc((size_t)capN * 4);
        P = pmpp::alloc((size_t)capPartials * 4);
        O = pmpp::alloc(16);
    }
    ~DeviceProblem() { pmpp::release(X); pmpp::release(P); pmpp::release(O); }
    DeviceProblem(const DeviceProblem&) = delete;
    void put(const std::vector<float>& x) { pmpp::upload(X, x.data(), x.size() * 4); }
    void poison(int partials) {   // outputs and partials start as NaN
        const float nan = std::numeric_limits<float>::quiet_NaN();
        std::vector<float> p((size_t)std::max(partials, 4), nan);
        pmpp::upload(P, p.data(), (size_t)std::min(partials, capPartials) * 4);
        pmpp::upload(O, p.data(), 16);
    }
    float result() { float r; pmpp::download(O, &r, 4); return r; }
};

static std::string fmtCase(Method m, int n, int coarse, int block, int launch) {
    char b[120];
    if (m == MSerial) std::snprintf(b, sizeof b, "n %d, launch %d", n, launch);
    else std::snprintf(b, sizeof b, "n %d, coarse %d, block %d, launch %d", n, coarse, block, launch);
    return b;
}

// ======================================================================= 6. tests
struct Sizes { std::vector<int> n, coarse, block; };
static Sizes testSizes() {
    const std::string b = pmpp::backend();
    if (b == "race") return {{33, 1000}, {1, 3}, {64}};
    if (b == "emu") return {{1, 2, 33, 1000, 4097}, {1, 3, 8}, {64, 256}};
    return {{1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003, 1 << 20}, {1, 2, 3, 8, 64}, {64, 256}};
}

// Prints each skip once.
static void noteSkip(const std::string& what) {
    static std::set<std::string> seen;
    if (seen.insert(what).second) std::printf("  skip: %s\n", what.c_str());
}

// Runs (method, acc) on x for every shape, each twice on the same buffers, and asks `judge`
// about every result. Returns the first failure, or "".
static std::string sweep(DeviceProblem& d, Method m, Acc acc, const std::vector<float>& x, const Sizes& sz,
                         const std::vector<int>& coarses,
                         const std::function<std::string(float got, int coarse, int block)>& judge) {
    const int n = (int)x.size();
    d.put(x);
    const std::vector<int> cs = m == MSerial ? std::vector<int>{1} : coarses;
    const std::vector<int> bs = m == MSerial ? std::vector<int>{1} : sz.block;
    for (int block : bs)
        for (int coarse : cs) {
            d.poison(m == MSerial ? 1 : blocksOf(n, coarse, block));
            for (int launch = 1; launch <= 2; ++launch) {
                Outcome o = runSum(m, acc, d.X, d.P, d.O, n, coarse, block);
                if (o.kind == Outcome::Skip) { noteSkip(o.why); break; }
                if (o.kind != Outcome::Ran) return o.why;
                const std::string err = pmpp::take_error();
                if (!err.empty()) return fmtCase(m, n, coarse, block, launch) + ": launch error: " + err;
                const std::string bad = judge(d.result(), coarse, block);
                if (!bad.empty()) return fmtCase(m, n, coarse, block, launch) + ": " + bad;
            }
        }
    return "";
}

static std::string show(double v) { char b[40]; std::snprintf(b, sizeof b, "%.9g", v); return b; }

static int tests(const pmpp::Args& a) {
    pmpp::Tests t(a);
    const Sizes sz = testSizes();
    const int maxN = *std::max_element(sz.n.begin(), sz.n.end());
    DeviceProblem d(std::max(maxN, 4097), 1);

    // The probe: describes the device (on the emulators, the host CPU).
    Probe pr = probe();
    std::string probeName = "device probe: rounding " + pr.rounding + ", denormals " +
                            (pr.keepsDenormals ? "kept" : "flushed") + ", a*b+c " + (pr.fusesMulAdd ? "fused" : "not fused");
    t.check(probeName, [&] { return pr.ok ? std::string() : pr.why; });

    for (Acc acc : {AccFloat, AccHalf, AccBfloat}) {
        std::string why;
        if (!accWanted(acc, &why)) { noteSkip(std::string(accName(acc)) + " accumulator: " + why); continue; }
        const Format& f = formatOf(acc);
        const bool narrow = acc != AccFloat;
        for (Method m : {MSerial, MTree, MKahan}) {
            std::vector<Set> sets = narrow ? std::vector<Set>{Set::Uniform, Set::Mixed, Set::Cancel}
                                           : std::vector<Set>{Set::Ones, Set::Ints, Set::Uniform, Set::Mixed, Set::Cancel};
            for (Set s : sets) {
                const bool exact = s == Set::Ones || s == Set::Ints;
                std::string name = std::string(methodName(m)) + ", " + accName(acc) + ", " + setName(s) +
                                   (exact ? ": exact" : ": within the bound");
                t.check(name, [&]() -> std::string {
                    for (int n : sz.n) {
                        const std::vector<float> x = makeData(s, n, acc);
                        const Ref ref = reference(x);
                        if (narrow && ref.sumAbs >= 65504.0) continue;   // keep every partial sum finite in FP16
                        std::string bad = sweep(d, m, acc, x, sz, sz.coarse, [&](float got, int coarse, int block) {
                            const double err = std::fabs((double)got - ref.sum);
                            if (exact) {
                                if (err == 0) return std::string();
                                return "got " + show(got) + ", the exact sum is " + show(ref.sum);
                            }
                            const double bound = relBound(m, acc, n, coarse, block) * ref.sumAbs + absAllowance(m, acc, n);
                            if (err <= bound) return std::string();
                            return "got " + show(got) + ", reference " + show(ref.sum) + ", error " + show(err) +
                                   " is above the bound " + show(bound);
                        });
                        if (!bad.empty()) return bad;
                    }
                    return std::string();
                });
            }
        }

        if (narrow) {
            // 4,097 ones: the result must have the bits of the CPU model of the same order, in
            // one of the two rounding modes a device may use. A kernel that accumulates in
            // float instead returns 4097.
            for (Method m : {MSerial, MTree}) {
                std::string name = std::string(methodName(m)) + ", " + accName(acc) + ", 4097 ones: the CPU model's result";
                t.check(name, [&]() -> std::string {
                    const std::vector<float> x = makeData(Set::Ones, 4097, acc);
                    return sweep(d, m, acc, x, sz, sz.coarse, [&](float got, int coarse, int block) {
                        const double rn = model(m, x, coarse, block, acc, Mode::Nearest);
                        const double rz = model(m, x, coarse, block, acc, Mode::TowardZero);
                        if ((double)got == rn || (double)got == rz) return std::string();
                        return "got " + show(got) + "; the CPU model gives " + show(rn) + " to nearest, " + show(rz) +
                               " toward zero";
                    });
                });
            }
        } else {
            // One big, many small, at coarse 64: thread 0 of block 0 adds 1 and 63 halves of an
            // ULP of 1. A plain loop drops every one of them (31.5 ULPs); the compensated loop
            // must keep them. This is the case that shows whether the compiler kept the
            // compensation.
            t.check("kahan, float, big-small at coarse 64: within 2 ULPs", [&]() -> std::string {
                for (int n : sz.n) {
                    if (n < 2) continue;
                    const std::vector<float> x = makeData(Set::BigSmall, n, acc);
                    const Ref ref = reference(x);
                    const double unit = fpformat::ulp(f, ref.sum);
                    std::string bad = sweep(d, MKahan, acc, x, sz, {64}, [&](float got, int, int) {
                        const double ulps = std::fabs((double)got - ref.sum) / unit;
                        if (ulps <= 2) return std::string();
                        return "got " + show(got) + ", reference " + show(ref.sum) + ": " + show(ulps) +
                               " ULPs off (a plain loop is 31.5 off where a thread holds 64 elements)";
                    });
                    if (!bad.empty()) return bad;
                }
                return std::string();
            });
        }
    }
    return t.finish();
}

// ======================================================================= 7. benchmark
static bool ranOrSkip(const Outcome& o, const std::string& label) {
    if (o.kind == Outcome::Ran) return true;
    std::printf("  %-28s skipped: %s\n", label.c_str(), o.why.c_str());
    return false;
}

static int bench(const pmpp::Args& a) {
    const Probe pr = probe();
    std::printf("probe (this device, default compiler settings): rounding %s, FP32 denormals %s, a*b+c %s\n",
                pr.ok ? pr.rounding.c_str() : "unknown", pr.keepsDenormals ? "kept" : "flushed",
                pr.fusesMulAdd ? "fused" : "not fused");
    pmpp::bench_row(a, "probe").s("rounding", pr.rounding).b("keeps_denormals", pr.keepsDenormals)
        .b("fuses_mul_add", pr.fusesMulAdd).emit();
    const Mode md = modeOf(pr);
    const int block = benchBlock(256);

    std::vector<Acc> accs;
    for (Acc acc : {AccFloat, AccHalf, AccBfloat}) {
        std::string why;
        if (accWanted(acc, &why)) accs.push_back(acc);
        else std::printf("%s accumulator skipped: %s\n", accName(acc), why.c_str());
    }

    // ---- accuracy: n = 65,536, coarse 16
    {
        const int n = 65536, coarse = 16;
        DeviceProblem d(n, block);
        std::printf("\naccuracy: n %d, coarse %d, block %d; ulps are in the accumulator's format at the reference;\n"
                    "model = same bits as the CPU model of that order, rounding %s\n",
                    n, coarse, block, fpformat::modeName(md));
        std::printf("  %-7s %-7s %-11s %15s %15s %11s %9s %6s\n", "method", "acc", "data", "result", "reference",
                    "rel err", "ulps", "model");
        const Set sets[] = {Set::Ones, Set::Uniform, Set::Mixed, Set::MixedAsc, Set::MixedDesc,
                            Set::Cancel, Set::Softmax, Set::BigSmall};
        for (Acc acc : accs)
            for (Method m : {MSerial, MTree, MKahan})
                for (Set s : sets) {
                    const std::vector<float> x = makeData(s, n, acc);
                    const Ref ref = reference(x);
                    d.put(x);
                    d.poison(blocksOf(n, coarse, block));
                    Outcome o = runSum(m, acc, d.X, d.P, d.O, n, coarse, block);
                    std::string label = std::string(methodName(m)) + " " + accName(acc) + " " + setName(s);
                    if (!ranOrSkip(o, label)) continue;
                    const float got = d.result();
                    const float mod = (float)model(m, x, coarse, block, acc, md);
                    const bool matches = std::memcmp(&got, &mod, 4) == 0 || (got == 0 && mod == 0);
                    const double absErr = std::fabs((double)got - ref.sum);
                    const double relErr = ref.sumAbs > 0 ? absErr / ref.sumAbs : 0;
                    const double ulps = absErr / fpformat::ulp(formatOf(acc), ref.sum);
                    std::printf("  %-7s %-7s %-11s %15.9g %15.9g %11.3g %9.3g %6s\n", methodName(m), accName(acc),
                                setName(s), got, ref.sum, relErr, ulps, matches ? "same" : "differs");
                    pmpp::bench_row(a, "accuracy").s("method", methodName(m)).s("acc", accName(acc)).s("data", setName(s))
                        .n("n", n).n("coarse", m == MSerial ? 1 : coarse).n("block", m == MSerial ? 1 : block)
                        .n("result", got).n("reference", ref.sum).n("sum_abs", ref.sumAbs).n("abs_err", absErr)
                        .n("rel_err", relErr).n("ulps", ulps).b("matches_model", matches)
                        .s("model_rounding", fpformat::modeName(md)).emit();
                }
    }

    // ---- coarse: tree and Kahan in FP32, 2^22 uniform floats
    {
        const int n = 1 << 22;
        DeviceProblem d(n, block);
        const std::vector<float> x = makeData(Set::Uniform, n, AccFloat);
        const Ref ref = reference(x);
        d.put(x);
        std::printf("\ncoarse: FP32, %d uniform floats, block %d, best of %d runs\n", n, block, a.reps);
        std::printf("  %-6s %6s %11s %9s %11s %9s\n", "method", "coarse", "time ms", "GB/s", "rel err", "ulps");
        for (Method m : {MTree, MKahan})
            for (int coarse : {1, 4, 16, 64, 256, 1024, 4096}) {
                Outcome first = runSum(m, AccFloat, d.X, d.P, d.O, n, coarse, block);
                if (!ranOrSkip(first, std::string(methodName(m)) + " coarse " + std::to_string(coarse))) continue;
                const double s = pmpp::time_it(a.reps, [&] { return runSum(m, AccFloat, d.X, d.P, d.O, n, coarse, block).time; }).min_s;
                const float got = d.result();
                const double absErr = std::fabs((double)got - ref.sum);
                const double relErr = absErr / ref.sumAbs;
                const double ulps = absErr / fpformat::ulp(fpformat::FP32, ref.sum);
                const double gbps = 4.0 * n / s / 1e9;
                std::printf("  %-6s %6d %11.4f %9.2f %11.3g %9.3g\n", methodName(m), coarse, s * 1e3, gbps, relErr, ulps);
                pmpp::bench_row(a, "coarse").s("method", methodName(m)).s("acc", "float").n("n", n).n("coarse", coarse)
                    .n("block", block).n("seconds", s).n("gb_per_s", gbps).n("rel_err", relErr).n("ulps", ulps).emit();
            }
    }

    // ---- time: tree and Kahan in each accumulator; serial at 2^20 only (one thread)
    {
        int n = 1 << 24;
        const double mem = deviceMemory();
        while (mem > 0 && n > (1 << 20) && 4.0 * n > mem / 8) n /= 2;   // stay well inside the device's memory
        const int nSerial = std::min(n, 1 << 20), coarse = 16;
        DeviceProblem d(n, block);
        std::printf("\ntime: uniform data, coarse %d, block %d, best of %d runs; GB/s = 4 bytes per input float / time\n",
                    coarse, block, a.reps);
        std::printf("  %-6s %-7s %9s %11s %9s\n", "method", "acc", "n", "time ms", "GB/s");
        for (Acc acc : accs) {
            const std::vector<float> x = makeData(Set::Uniform, n, acc);   // serial uses the first nSerial
            d.put(x);
            for (Method m : {MSerial, MTree, MKahan}) {
                const int nn = m == MSerial ? nSerial : n;
                Outcome first = runSum(m, acc, d.X, d.P, d.O, nn, coarse, block);
                if (!ranOrSkip(first, std::string(methodName(m)) + " " + accName(acc))) continue;
                const double s = pmpp::time_it(a.reps, [&] { return runSum(m, acc, d.X, d.P, d.O, nn, coarse, block).time; }).min_s;
                const double gbps = 4.0 * nn / s / 1e9;
                std::printf("  %-6s %-7s %9d %11.4f %9.2f\n", methodName(m), accName(acc), nn, s * 1e3, gbps);
                pmpp::bench_row(a, "time").s("method", methodName(m)).s("acc", accName(acc)).n("n", nn)
                    .n("coarse", m == MSerial ? 1 : coarse).n("block", m == MSerial ? 1 : block)
                    .n("seconds", s).n("gb_per_s", gbps).emit();
            }
        }
    }
    const std::string err = pmpp::take_error();
    if (!err.empty()) { std::printf("error during the benchmark: %s\n", err.c_str()); return 1; }
    std::printf("\nThese are measurements of this machine at these sizes, not properties of the kernels.\n");
    return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
