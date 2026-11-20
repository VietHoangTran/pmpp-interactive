// Lab 3.3 driver: tests and benchmarks the naive GEMM (one thread per element
// of C), the first half of project milestone M1.
// You do not need to edit this file; your work goes in work/.
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// Both emulator flavours run on the CPU and get the small problems only.
#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)
static const bool kEmulated = true;
#else
static const bool kEmulated = false;
#endif

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeNaiveGemm(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A, id<MTLBuffer> B,
                     id<MTLBuffer> C, uint32_t j, uint32_t k, uint32_t l);
static id<MTLComputePipelineState> gemmPipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "naive_gemm"); }
static pmpp::Timing runGemm(Buf* A, Buf* B, Buf* C, int j, int k, int l) {
  auto pso = gemmPipeline();
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeNaiveGemm(((pmpp::metal::Pass*)v)->encoder, pso, pmpp::metal::buffer(A), pmpp::metal::buffer(B),
                    pmpp::metal::buffer(C), (uint32_t)j, (uint32_t)k, (uint32_t)l);
  });
}
// The lab fixes the threadgroup at 16 x 16. Metal's limit belongs to the pipeline,
// not to the device, so ask the pipeline. Empty string = fine.
static std::string launchShapeProblem() {
  auto pso = gemmPipeline();
  if (!pso || pso.maxTotalThreadsPerThreadgroup >= 256) return "";
  return "this pipeline allows " + std::to_string(pso.maxTotalThreadsPerThreadgroup) +
         " threads per threadgroup, fewer than the 16 x 16 the lab uses";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchNaiveGemm(float* A, float* B, float* C, int j, int k, int l);
static pmpp::Timing runGemm(Buf* A, Buf* B, Buf* C, int j, int k, int l) {
  return pmpp::run([&](void*) {
    launchNaiveGemm((float*)pmpp::raw(A), (float*)pmpp::raw(B), (float*)pmpp::raw(C), j, k, l);
  });
}
static std::string launchShapeProblem() {
  const double most = pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (most >= 256) return "";
  return "this device allows " + std::to_string((int)most) + " threads per block, fewer than the 16 x 16 the lab uses";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------- what the tests put around one multiplication ----------
// C is pre-filled with a value no product equals, so an element the kernel never
// writes is detected.
//   Emulator (backend emu): every buffer has exactly its size, and AddressSanitizer
//   stops the run at the first access outside it and names the line.
//   Everywhere else: nothing reports such an access, so every buffer gets slack after
//   its last element. The slack after A and B holds NaN (a kernel that reads it
//   produces NaN); the slack after C holds a canary (a kernel that writes it is caught).
static const float kUnwritten = 12345.0f;
static const float kCanary = -54321.0f;

struct Guarded {
  int j, k, l;
  size_t slack;   // floats after the last element of each buffer
  Buf *A, *B, *C;

  Guarded(const std::vector<float>& a, const std::vector<float>& b, int j_, int k_, int l_) : j(j_), k(k_), l(l_) {
    const size_t m = (size_t)std::max(j, std::max(k, l));
    slack = pmpp::is_emu() ? 0 : m * m + 32 * m + 64;
    A = padded(a, std::numeric_limits<float>::quiet_NaN());
    B = padded(b, std::numeric_limits<float>::quiet_NaN());
    std::vector<float> c((size_t)j * l, kUnwritten);
    C = padded(c, kCanary);
  }
  Guarded(const Guarded&) = delete;
  ~Guarded() { pmpp::release(A); pmpp::release(B); pmpp::release(C); }

  Buf* padded(const std::vector<float>& v, float fill) const {
    std::vector<float> host(v);
    host.resize(v.size() + slack, fill);
    Buf* b = pmpp::alloc(host.size() * 4);
    pmpp::upload(b, host.data(), host.size() * 4);
    return b;
  }

  // Runs the kernel. Empty string = C matches `ref` and nothing outside C was written.
  std::string run(const std::vector<float>& ref, const std::vector<float>& mag) {
    runGemm(A, B, C, j, k, l);
    const size_t n = (size_t)j * l;
    std::vector<float> got(n + slack);
    pmpp::download(C, got.data(), got.size() * 4);
    std::string msg = pmpp::describe(pmpp::compare(got.data(), ref.data(), mag.data(), n), l);
    size_t unwritten = 0, past = 0;
    for (size_t i = 0; i < n; ++i) unwritten += (got[i] == kUnwritten);
    for (size_t i = n; i < got.size(); ++i) past += !(got[i] == kCanary);
    auto add = [&](const std::string& s) { msg += (msg.empty() ? "" : "; ") + s; };
    if (unwritten) add(std::to_string(unwritten) + " of " + std::to_string(n) + " elements of C were never written");
    if (past) add(std::to_string(past) + " values were stored past the end of C");
    return msg;
  }
};

// y = x W^T, with W as PyTorch stores a linear layer's weight: (out_features,
// in_features), row-major. Row b of x and row o of W are both read along a row.
static void cpu_linear(const std::vector<float>& x, const std::vector<float>& W, int batch, int in, int out,
                       std::vector<float>& y, std::vector<float>& mag) {
  y.assign((size_t)batch * out, 0.f);
  mag.assign((size_t)batch * out, 0.f);
  for (int b = 0; b < batch; ++b)
    for (int o = 0; o < out; ++o) {
      double acc = 0, am = 0;
      for (int i = 0; i < in; ++i) {
        const double p = (double)x[(size_t)b * in + i] * W[(size_t)o * in + i];
        acc += p; am += std::fabs(p);
      }
      y[(size_t)b * out + o] = (float)acc;
      mag[(size_t)b * out + o] = (float)am;
    }
}

struct Shape { int j, k, l; const char* why; };

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string shapeProblem = launchShapeProblem();
  if (!shapeProblem.empty()) t.check("16 x 16 launch usable here", [&] { return shapeProblem; });

  std::vector<Shape> shapes = {
      {4, 4, 4, "the size of the book's Fig. 3.12"},
      {16, 16, 16, "exactly one block"},
      {33, 33, 33, "square, one past two blocks"},
      {64, 64, 64, "square, whole blocks"},
      {5, 7, 3, "small and rectangular"},
      {1, 40, 33, "one row through the layer: the decode shape"},
      {33, 20, 1, "one output column: matrix times vector, book exercise 2"},
      {20, 1, 35, "inner dimension 1: an outer product"},
      {17, 40, 33, "nothing is a multiple of the block"},
  };
  if (!kEmulated) shapes.push_back({256, 1024, 512, "a batch of 256 through a 1024 to 512 layer"});

  for (const Shape& s : shapes) {
    char name[160];
    std::snprintf(name, sizeof name, "%d x %d times %d x %d  (%s)", s.j, s.k, s.k, s.l, s.why);
    t.check(name, [&]() -> std::string {
      const uint32_t seed = 3000 + s.j * 31 + s.k * 7 + s.l;
      auto av = pmpp::random_floats((size_t)s.j * s.k, seed);
      auto bv = pmpp::random_floats((size_t)s.k * s.l, seed + 1);
      std::vector<float> ref, mag;
      pmpp::cpu_gemm(av.data(), bv.data(), s.j, s.k, s.l, ref, &mag);
      Guarded g(av, bv, s.j, s.k, s.l);
      return g.run(ref, mag);
    });
  }

  // The same kernel as a linear layer. The weight is built the way PyTorch stores
  // it, (out, in), transposed on the host into the (in, out) matrix this kernel
  // takes as B, and the result is checked against y = x W^T computed from W itself.
  struct Layer { int batch, in, out; };
  std::vector<Layer> layers = {{5, 48, 20}};
  if (!kEmulated) layers.push_back({3, 300, 1000});
  for (const Layer& L : layers) {
    char name[200];
    std::snprintf(name, sizeof name,
                  "linear layer, batch %d, %d inputs, %d outputs: weight stored (out, in), transposed on the host, y = x W^T",
                  L.batch, L.in, L.out);
    t.check(name, [&]() -> std::string {
      auto x = pmpp::random_floats((size_t)L.batch * L.in, 77 + L.in);
      auto W = pmpp::random_floats((size_t)L.out * L.in, 78 + L.out);
      std::vector<float> Bt((size_t)L.in * L.out);
      for (int o = 0; o < L.out; ++o)
        for (int i = 0; i < L.in; ++i) Bt[(size_t)i * L.out + o] = W[(size_t)o * L.in + i];
      std::vector<float> y, mag;
      cpu_linear(x, W, L.batch, L.in, L.out, y, mag);
      Guarded g(x, Bt, L.batch, L.in, L.out);
      return g.run(y, mag);
    });
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // Skip a problem whose three matrices would need more than a quarter of the
  // memory this device reports.
  const double mem = deviceMemory();
  auto fits = [&](double j, double k, double l) { return mem <= 0 || 4.0 * (j * k + k * l + j * l) <= mem / 4; };

  std::printf("naive GEMM, 16 x 16 blocks, best of %d runs\n", a.reps);
  std::printf("square matrices\n  %6s %12s %14s %24s\n", "width", "time (ms)", "GFLOP/s", "time / time at half width");
  double prev = 0;
  for (int w : {256, 512, 1024}) {
    if (!fits(w, w, w)) { std::printf("  %6d skipped: does not fit in a quarter of the device's memory\n", w); prev = 0; continue; }
    pmpp::GemmProblem p(w, w, w, 5, false);
    const double s = pmpp::time_it(a.reps, [&] { return runGemm(p.A, p.B, p.C, w, w, w); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  failed: %s\n", err.c_str()); return 1; }
    pmpp::Json row = pmpp::bench_row(a, "square");
    row.n("width", w).n("seconds", s).n("gflops", p.flop() / s / 1e9);
    if (prev > 0) {
      std::printf("  %6d %12.4f %14.2f %24.2f\n", w, s * 1e3, p.flop() / s / 1e9, s / prev);
      row.n("time_vs_half_width", s / prev);
    } else {
      std::printf("  %6d %12.4f %14.2f %24s\n", w, s * 1e3, p.flop() / s / 1e9, "");
    }
    row.emit();
    prev = s;
  }
  std::printf("  (each doubling of the width is 8 times the arithmetic)\n");

  // The layer lab 5.3 times: 1024 inputs, 4096 outputs. The batch size j is the
  // number of rows pushed through it at once. Same row name and keys as lab 5.3,
  // so the two milestones can be put side by side.
  const int k = 1024, l = 4096;
  std::printf("linear layer %d -> %d\n  %6s %12s %14s %16s\n", k, l, "batch", "time (ms)", "GFLOP/s", "rows per second");
  for (int j : {1, 16, 256}) {
    if (!fits(j, k, l)) { std::printf("  %6d skipped: does not fit in a quarter of the device's memory\n", j); continue; }
    pmpp::GemmProblem p(j, k, l, 3, false);
    const double s = pmpp::time_it(a.reps, [&] { return runGemm(p.A, p.B, p.C, j, k, l); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  failed: %s\n", err.c_str()); return 1; }
    std::printf("  %6d %12.4f %14.2f %16.0f\n", j, s * 1e3, p.flop() / s / 1e9, j / s);
    pmpp::bench_row(a, "linear_layer").n("batch", j).n("k", k).n("l", l).n("block", 16)
        .n("seconds", s).n("gflops", p.flop() / s / 1e9).n("rows_per_s", j / s).emit();
  }
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
