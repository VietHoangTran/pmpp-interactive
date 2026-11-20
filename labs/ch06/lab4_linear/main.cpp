// Lab 6.4 driver: tests your corner-turned linear layer and the provided
// uncoalesced baseline against a CPU reference, then times both.
//   Y[batch x out] = X[batch x in] * W^T, W stored (out, in) row-major.
// You do not need to edit this file; your work goes in work/.
#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
using pmpp::Buf;

// ---------------------------------------------------------------- the problem
// Y[b][o] = sum_k X[b][k] * W[o][k], accumulated in double. `mag` receives
// sum_k |X[b][k] * W[o][k]| per output, which scales the tolerance.
static void cpu_linear(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
                       std::vector<float>& y, std::vector<float>& mag) {
  y.assign((size_t)batch * out, 0.f);
  mag.assign((size_t)batch * out, 0.f);
  for (int b = 0; b < batch; ++b) {
    const float* xr = x.data() + (size_t)b * in;
    for (int o = 0; o < out; ++o) {
      const float* wr = w.data() + (size_t)o * in;
      double acc = 0, am = 0;
      for (int k = 0; k < in; ++k) { const double p = (double)xr[k] * wr[k]; acc += p; am += std::fabs(p); }
      y[(size_t)b * out + o] = (float)acc;
      mag[(size_t)b * out + o] = (float)am;
    }
  }
}

struct LinearProblem {
  int batch, in, out;
  std::vector<float> x, w, ref, mag;
  Buf *X = nullptr, *W = nullptr, *Y = nullptr;
  LinearProblem(int b, int i, int o, uint32_t seed, bool with_data = true) : batch(b), in(i), out(o) {
    X = pmpp::alloc((size_t)batch * in * 4);
    W = pmpp::alloc((size_t)out * in * 4);
    Y = pmpp::alloc((size_t)batch * out * 4);
    if (with_data) {
      x = pmpp::random_floats((size_t)batch * in, seed);
      w = pmpp::random_floats((size_t)out * in, seed + 1);
      pmpp::upload(X, x.data(), x.size() * 4);
      pmpp::upload(W, w.data(), w.size() * 4);
    }
  }
  LinearProblem(const LinearProblem&) = delete;
  ~LinearProblem() { pmpp::release(X); pmpp::release(W); pmpp::release(Y); }
  double flop() const { return 2.0 * batch * (double)in * out; }
  // A value no correct kernel produces here, so elements never written are detected.
  void poison() {
    std::vector<float> p((size_t)batch * out, 12345.0f);
    pmpp::upload(Y, p.data(), p.size() * 4);
  }
  std::string check() {
    if (ref.empty()) cpu_linear(x, w, batch, in, out, ref, mag);
    std::vector<float> got((size_t)batch * out);
    pmpp::download(Y, got.data(), got.size() * 4);
    return pmpp::describe(pmpp::compare(got.data(), ref.data(), mag.data(), got.size()), out);
  }
};

// ---------------------------------------------------------------- backends
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeLinear(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                  id<MTLBuffer> Y, int32_t batch, int32_t in, int32_t out, int32_t tileWidth);
void encodeLinearBaseline(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X,
                          id<MTLBuffer> W, id<MTLBuffer> Y, int32_t batch, int32_t in, int32_t out,
                          int32_t tileWidth);
using pmpp::metal::buffer;

// provided/linear_baseline.metal, found from this variant's kernel.metal: <lab>/<variant>/kernel.metal.
static std::string providedMetal() {
  std::string s = PMPP_METAL_SOURCE;
  s = s.substr(0, s.rfind('/'));
  s = s.substr(0, s.rfind('/'));
  return s + "/provided/linear_baseline.metal";
}
static id<MTLComputePipelineState> pipelineFor(bool baseline, int tile) {
  const std::vector<std::pair<std::string, std::string>> m = {{"TILE_WIDTH", std::to_string(tile)}};
  return baseline ? pmpp::metal::pipeline(providedMetal(), "linear_baseline", m)
                  : pmpp::metal::pipeline(PMPP_METAL_SOURCE, "linear", m);
}
// Empty if the kernel can run with tile x tile threads per threadgroup, else why not.
static std::string tileProblem(bool baseline, int tile) {
  id<MTLComputePipelineState> p = pipelineFor(baseline, tile);
  if (!p) return "the kernel did not compile: " + pmpp::take_error();
  if ((NSUInteger)(tile * tile) > p.maxTotalThreadsPerThreadgroup)
    return "this pipeline allows at most " + std::to_string((unsigned long)p.maxTotalThreadsPerThreadgroup) +
           " threads per threadgroup";
  return "";
}
static pmpp::Timing runLinear(LinearProblem& p, bool baseline, int tile) {
  id<MTLComputePipelineState> pso = pipelineFor(baseline, tile);
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    auto* pass = (pmpp::metal::Pass*)v;
    if (baseline)
      encodeLinearBaseline(pass->encoder, pso, buffer(p.X), buffer(p.W), buffer(p.Y), p.batch, p.in, p.out, tile);
    else
      encodeLinear(pass->encoder, pso, buffer(p.X), buffer(p.W), buffer(p.Y), p.batch, p.in, p.out, tile);
  });
}
// Metal compiles the tile width at run time, so one executable covers both.
static std::vector<int> tiles() { return {16, 32}; }
#else
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out);
void launchLinearBaseline(const float* X, const float* W, float* Y, int batch, int in, int out);
static std::string tileProblem(bool, int tile) {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (tile * tile > limit) return "this device allows at most " + std::to_string(limit) + " threads per block";
  return "";
}
static pmpp::Timing runLinear(LinearProblem& p, bool baseline, int /*tile: fixed at compile time*/) {
  const float* X = (const float*)pmpp::raw(p.X);
  const float* W = (const float*)pmpp::raw(p.W);
  float* Y = (float*)pmpp::raw(p.Y);
  return pmpp::run([&](void*) {
    if (baseline) launchLinearBaseline(X, W, Y, p.batch, p.in, p.out);
    else launchLinear(X, W, Y, p.batch, p.in, p.out);
  });
}
// CUDA, HIP and the emulator compile TILE_WIDTH in; the t32 build times T = 32.
static std::vector<int> tiles() { return {TILE_WIDTH}; }
#endif

static const char* who(bool baseline) { return baseline ? "provided baseline" : "your kernel"; }

// ---------------------------------------------------------------- tests
struct Shape { int batch, in, out; const char* why; };

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::vector<Shape> shapes = {
      {3, 3, 3, "smaller than one tile"},
      {1, 1, 1, "single element"},
      {16, 16, 16, "exactly one tile at T = 16"},
      {32, 48, 16, "rectangular, multiples of 16"},
      {17, 33, 9, "nothing is a multiple of the tile"},
      {1, 40, 7, "one token"},
      {5, 1, 9, "one input feature"},
      {33, 65, 50, "partial tiles in every direction"},
  };
  if (!pmpp::is_emu()) {
    shapes.push_back({4, 300, 1000, "a small batch into a wide layer"});
    shapes.push_back({257, 129, 511, "large and awkward"});
    shapes.push_back({1, 1024, 4096, "one token through the benchmark layer"});
  }
  for (int tile : tiles()) {
    for (bool baseline : {false, true}) {
      const std::string problem = tileProblem(baseline, tile);
      if (!problem.empty()) {
        // The default tile must work; a larger one may exceed what this device allows.
        if (tile == TILE_WIDTH || problem.rfind("the kernel did not compile", 0) == 0)
          t.check(std::string(who(baseline)) + ", tile " + std::to_string(tile) + " runs here",
                  [&] { return problem; });
        else
          std::printf("  note: %s, tile %d not tested: %s\n", who(baseline), tile, problem.c_str());
        continue;
      }
      for (const Shape& s : shapes) {
        char name[200];
        std::snprintf(name, sizeof name, "%s, tile %d: batch %d, in %d, out %d  (%s)", who(baseline), tile,
                      s.batch, s.in, s.out, s.why);
        t.check(name, [&]() -> std::string {
          LinearProblem p(s.batch, s.in, s.out, 2000 + s.batch * 31 + s.in * 7 + s.out);
          p.poison();
          runLinear(p, baseline, tile);
          return p.check();
        });
      }
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// One row per timed kernel. The kernel goes in "variant" (the field the lesson
// pages read) and again in "kernel": the verify report's tables leave "variant"
// out, because there it means work or solution, and the result holding these rows
// already records which of the two ran.
static pmpp::Json row(const pmpp::Args& a, const char* kernel) {
  pmpp::Json j;
  j.s("kind", "bench").s("lab", a.lab).s("backend", pmpp::backend()).s("name", "linear")
   .s("variant", kernel).s("kernel", kernel);
  return j;
}

// Best time of the baseline and of your kernel over `reps` rounds, after one
// warm-up of each. The two take turns within a round, so a change of clock speed,
// or other work on the machine, during the measurement affects both alike.
static std::pair<double, double> timePair(int reps, LinearProblem& p, int tile) {
  runLinear(p, true, tile);
  runLinear(p, false, tile);
  double sb = 0, st = 0;
  for (int r = 0; r < reps; ++r) {
    const double b = runLinear(p, true, tile).best(), t = runLinear(p, false, tile).best();
    if (r == 0 || b < sb) sb = b;
    if (r == 0 || t < st) st = t;
  }
  return {sb, st};
}

static int bench(const pmpp::Args& a) {
  // Lab 5.3's layer, so the two labs can be compared: 1024 inputs -> 4096 outputs.
  const int in = 1024, out = 4096;
  std::printf("linear layer %d -> %d, W stored (out, in); best of %d runs after a warm-up,\n"
              "the two kernels taking turns\n", in, out, a.reps);
  int rc = 0;
  for (int tile : tiles()) {
    const std::string problem = tileProblem(false, tile) + tileProblem(true, tile);
    if (!problem.empty()) { std::printf("\ntile %d skipped: %s\n", tile, problem.c_str()); continue; }
    // A timing of a wrong kernel means nothing: check both on an awkward shape first.
    // (Builds with an extra tile width are benchmarked but not run through the tests.)
    for (bool baseline : {true, false}) {
      LinearProblem p(33, 65, 50, 77);
      p.poison();
      runLinear(p, baseline, tile);
      std::string err = p.check();
      if (!err.empty()) {
        std::printf("\ntile %d: %s gives wrong results (%s); not timed\n", tile, who(baseline), err.c_str());
        rc = 1;
      }
    }
    if (rc) continue;
    std::printf("\ntile %d\n  %6s %16s %10s %16s %10s %10s\n", tile, "batch", "baseline (ms)", "GFLOP/s",
                "yours (ms)", "GFLOP/s", "speed-up");
    for (int batch : {1, 16, 64, 256, 1024}) {
      LinearProblem p(batch, in, out, 3, false);
      const auto [sb, st] = timePair(a.reps, p, tile);
      const std::string err = pmpp::take_error();
      if (!err.empty() || sb <= 0 || st <= 0) {
        std::printf("  %6d failed: %s\n", batch, err.empty() ? "no time reported" : err.c_str());
        rc = 1;
        break;
      }
      const double gb = p.flop() / sb / 1e9, gt = p.flop() / st / 1e9;
      std::printf("  %6d %16.4f %10.1f %16.4f %10.1f %9.2fx\n", batch, sb * 1e3, gb, st * 1e3, gt, sb / st);
      row(a, "baseline").n("tile", tile).n("batch", batch).n("in", in).n("out", out)
          .n("seconds", sb).n("gflops", gb).emit();
      row(a, "tuned").n("tile", tile).n("batch", batch).n("in", in).n("out", out)
          .n("seconds", st).n("gflops", gt).n("speedup_vs_baseline", sb / st).emit();
    }
  }
  std::printf("\nThese are measurements of this machine, not expectations.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
