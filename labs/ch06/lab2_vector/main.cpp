// Lab 6.2 driver: tests and benchmarks vector addition with float4 accesses
// (vecAdd4Kernel) and with thread coarsening (vecAddCoarseKernel), against the
// provided one-element-per-thread kernel. You do not need to edit this file;
// your work goes in work/.
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeVecAdd(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                  id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n);
void encodeVecAdd4(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                   id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n);
void encodeVecAddCoarse(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                        id<MTLBuffer> x, id<MTLBuffer> y, id<MTLBuffer> z, int32_t n);
using pmpp::metal::buffer;
using Encoder = void (*)(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                         id<MTLBuffer>, id<MTLBuffer>, id<MTLBuffer>, int32_t);
static std::string dirOf(const std::string& p) { return p.substr(0, p.rfind('/')); }
static pmpp::Timing runMetal(const std::string& path, const char* fn, Encoder enc, Buf* x, Buf* y, Buf* z, int n) {
  id<MTLComputePipelineState> p = pmpp::metal::pipeline(path, fn);
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) { enc(((pmpp::metal::Pass*)v)->encoder, p, buffer(x), buffer(y), buffer(z), n); });
}
static pmpp::Timing runScalar(Buf* x, Buf* y, Buf* z, int n) {
  return runMetal(dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/vecadd.metal", "vec_add", encodeVecAdd, x, y, z, n);
}
static pmpp::Timing runVec4(Buf* x, Buf* y, Buf* z, int n) {
  return runMetal(PMPP_METAL_SOURCE, "vec_add4", encodeVecAdd4, x, y, z, n);
}
static pmpp::Timing runCoarse(Buf* x, Buf* y, Buf* z, int n) {
  return runMetal(PMPP_METAL_SOURCE, "vec_add_coarse", encodeVecAddCoarse, x, y, z, n);
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchVecAdd(const float* x, const float* y, float* z, int n);
void launchVecAdd4(const float* x, const float* y, float* z, int n);
void launchVecAddCoarse(const float* x, const float* y, float* z, int n);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static pmpp::Timing runScalar(Buf* x, Buf* y, Buf* z, int n) { return pmpp::run([&](void*) { launchVecAdd(f(x), f(y), f(z), n); }); }
static pmpp::Timing runVec4(Buf* x, Buf* y, Buf* z, int n) { return pmpp::run([&](void*) { launchVecAdd4(f(x), f(y), f(z), n); }); }
static pmpp::Timing runCoarse(Buf* x, Buf* y, Buf* z, int n) { return pmpp::run([&](void*) { launchVecAddCoarse(f(x), f(y), f(z), n); }); }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

using Runner = pmpp::Timing (*)(Buf*, Buf*, Buf*, int);
struct Kernel { const char* key; const char* label; Runner run; };
// key: the value of "variant" in the benchmark rows.
static const Kernel kScalar = {"scalar", "provided, one float per thread", runScalar};
static const Kernel kCoarse = {"coarse4", "vecAddCoarseKernel, 4 floats per thread", runCoarse};
static const Kernel kVec4 = {"vec4", "vecAdd4Kernel, float4 accesses", runVec4};

// Three device arrays of exactly n floats each (no slack: on the emulator a float4
// access past the end is an AddressSanitizer error). z starts as NaN, so an element
// your kernel never writes is reported.
struct VecProblem {
  int n;
  std::vector<float> x, y;
  Buf *X, *Y, *Z;
  VecProblem(int n_, uint32_t seed) : n(n_) {
    x = pmpp::random_floats(n, seed);
    y = pmpp::random_floats(n, seed + 1);
    X = pmpp::alloc((size_t)n * 4); Y = pmpp::alloc((size_t)n * 4); Z = pmpp::alloc((size_t)n * 4);
    pmpp::upload(X, x.data(), (size_t)n * 4);
    pmpp::upload(Y, y.data(), (size_t)n * 4);
    std::vector<float> poison(n, std::numeric_limits<float>::quiet_NaN());
    pmpp::upload(Z, poison.data(), (size_t)n * 4);
  }
  VecProblem(const VecProblem&) = delete;
  ~VecProblem() { pmpp::release(X); pmpp::release(Y); pmpp::release(Z); }

  std::string check() const {
    std::vector<float> got(n), ref(n);
    pmpp::download(Z, got.data(), (size_t)n * 4);
    for (int i = 0; i < n; ++i) ref[i] = x[i] + y[i];
    pmpp::Cmp c = pmpp::compare(got.data(), ref.data(), nullptr, (size_t)n, 1e-6);
    if (c.ok) return "";
    const size_t i = c.first_bad;
    const int fullEnd = n - n % 4;   // elements from here on are in the partial last chunk
    char b[240];
    std::snprintf(b, sizeof b, "%zu of %d elements wrong; first at index %zu (%s), got %g, expected %g",
                  c.bad, n, i, (int)i >= fullEnd ? "in the partial last chunk" : "in a full chunk",
                  (double)got[i], (double)ref[i]);
    return b;
  }
};

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::vector<int> sizes = {1, 2, 3, 4, 5, 7, 8, 9, 255, 256, 257, 1023, 1024, 1025};
  if (!pmpp::is_emu()) {
    sizes.push_back(1 << 20);
    sizes.push_back((1 << 20) + 3);
    sizes.push_back(10000001);
  }
  t.check("provided baseline, n = 1025", [&]() -> std::string {
    VecProblem p(1025, 11);
    runScalar(p.X, p.Y, p.Z, p.n);
    return p.check();
  });
  for (const Kernel* k : {&kVec4, &kCoarse}) {
    for (int n : sizes) {
      char name[120];
      std::snprintf(name, sizeof name, "%s, n = %d (n mod 4 = %d)", k->key, n, n % 4);
      t.check(name, [&]() -> std::string {
        VecProblem p(n, 100 + (uint32_t)n % 1000);
        k->run(p.X, p.Y, p.Z, n);
        return p.check();
      });
    }
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // Three arrays of n floats. Grow n from 2^22 until the baseline takes long enough to
  // time reliably, up to 2^26, and keep the arrays within a quarter of the memory the
  // device reports.
  const double mem = deviceMemory();
  auto fits = [&](long long n) { return mem <= 0 || 12.0 * (double)n <= mem / 4; };
  int n = 1 << 22;
  for (;;) {
    VecProblem p(n, 7);
    const double s = pmpp::time_it(a.reps, [&] { return runScalar(p.X, p.Y, p.Z, n); }).min_s;
    if (s > 0.005 || n >= (1 << 26) || !fits(2LL * n)) break;
    n *= 2;
  }
  VecProblem p(n, 7);
  const double bytes = 12.0 * n;   // two 4-byte loads and one 4-byte store per element
  std::printf("n = %d floats per array, %.0f MB moved per addition, best of %d runs\n",
              n, bytes / 1e6, a.reps);
  std::printf("  %-8s %-40s %10s %9s %10s\n", "variant", "kernel", "time ms", "GB/s", "vs scalar");
  double scalar_s = 0;
  for (const Kernel* k : {&kScalar, &kCoarse, &kVec4}) {
    const double s = pmpp::time_it(a.reps, [&] { return k->run(p.X, p.Y, p.Z, n); }).min_s;
    const std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) {
      std::printf("  %-8s %-40s failed: %s\n", k->key, k->label, err.c_str());
      continue;
    }
    if (k == &kScalar) scalar_s = s;
    const double speedup = scalar_s > 0 ? scalar_s / s : std::nan("");
    std::printf("  %-8s %-40s %10.4f %9.1f %9.2fx\n", k->key, k->label, s * 1e3, bytes / s / 1e9, speedup);
    // The row's "variant" names the kernel; the work/solution variant is on the run record.
    pmpp::Json j;
    j.s("kind", "bench").s("lab", a.lab).s("backend", pmpp::backend()).s("name", "vecadd")
     .s("variant", k->key).n("n", n).n("seconds", s).n("gb_per_s", bytes / s / 1e9)
     .n("speedup_vs_scalar", speedup).emit();
  }
  std::printf("These are measurements of three kernels on this machine at one size.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
