// Lab 15.6 driver (milestone M3): tests your fast linear layer against a CPU reference,
// at warp widths 32 and 64, then times it beside lab 6.4's M2 kernel and the vendor
// library.
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major.
// You do not need to edit this file; your work goes in work/.
#include "../shared/linear_test.hpp"

#include <algorithm>
#include <cmath>
#include <memory>

using lin::Case;
using lin::Shape;
using pmpp::Buf;

// ---------------------------------------------------------------- the CPU reference
// Y[b][o] = sum_k X[b][k] * W[o][k], accumulated in double; mag = sum_k |X[b][k] * W[o][k]|,
// which scales the tolerance. Declared in ../shared/linear_test.hpp, whose test cases call it.
void lin::cpuLinear(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
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

// ---------------------------------------------------------------- backends
// Each backend section defines:
//   m3Problem(width)   empty if your kernel, built for that warp width, can run here
//   runM3(c, width)    your kernel at a given width (every width gives the same Y)
//   deviceWidth()      the width this device runs warps at, and where it came from
//   runM3Device(c)     your kernel the way the project calls it (device's width)
//   m2Problem / runM2, vendorProblem / runVendor: the two baselines
//   kernelRow(a, w)    the occupancy row (CUDA, HIP) or the pipeline row (Metal)
const char* vendorName();    // provided/vendor_linear.cu or provided/vendor_linear_mps.mm
std::string vendorMath();
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeLinear(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                  id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
bool encodeLinearVendor(pmpp::metal::Pass*, id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y, uint32_t batch,
                        uint32_t in, uint32_t out);                         // provided/vendor_linear_mps.mm
#if defined(PMPP_M2_METAL)
void encodeLinearM2(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                    id<MTLBuffer> Y, int32_t batch, int32_t in, int32_t out, int32_t tileWidth);   // lab 6.4's
#endif
using pmpp::metal::buffer;

static id<MTLComputePipelineState> m3Pipeline(int width) {
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "linear_m3", {{"SIMD_WIDTH", std::to_string(width)}});
}
// Empty if the pipeline exists and can run `threads` threads per threadgroup here.
static std::string pipelineProblem(id<MTLComputePipelineState> p, NSUInteger threads, const std::string& what) {
  char b[300];
  if (!p) return what + " did not compile: " + pmpp::take_error();
  if (p.maxTotalThreadsPerThreadgroup < threads) {
    std::snprintf(b, sizeof b, "%s: this pipeline allows %lu threads per threadgroup, the kernel needs %lu",
                  what.c_str(), (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)threads);
    return b;
  }
  const NSUInteger have = pmpp::metal::device().maxThreadgroupMemoryLength;
  if (p.staticThreadgroupMemoryLength > have) {
    std::snprintf(b, sizeof b, "%s: the tiles take %lu bytes of threadgroup memory, this device has %lu", what.c_str(),
                  (unsigned long)p.staticThreadgroupMemoryLength, (unsigned long)have);
    return b;
  }
  return "";
}
static std::string m3Problem(int width) {
  return pipelineProblem(m3Pipeline(width), 256, "linear_m3 with SIMD_WIDTH=" + std::to_string(width));
}
static pmpp::Timing runM3(Case& c, int width) {
  id<MTLComputePipelineState> p = m3Pipeline(width);
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeLinear(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), buffer(c.W), buffer(c.Y), (uint32_t)c.batch,
                 (uint32_t)c.in, (uint32_t)c.out);
  });
}
// Apple documents no fixed SIMD-group width; the pipeline reports it (ch15-facts C1).
static int reportedWidth() {
  id<MTLComputePipelineState> p = m3Pipeline(32);
  return p ? (int)p.threadExecutionWidth : 0;
}
static int deviceWidth() { return reportedWidth() == 64 ? 64 : 32; }
static std::string widthSource() {
  return "threadExecutionWidth of the linear_m3 pipeline: " + std::to_string(reportedWidth());
}
static pmpp::Timing runM3Device(Case& c) { return runM3(c, deviceWidth()); }

#if defined(PMPP_M2_METAL)
static id<MTLComputePipelineState> m2Pipeline() {
  return pmpp::metal::pipeline(PMPP_M2_METAL, "linear", {{"TILE_WIDTH", "16"}});
}
static std::string m2Problem() { return pipelineProblem(m2Pipeline(), 256, "lab 6.4's linear kernel"); }
static pmpp::Timing runM2(Case& c) {
  id<MTLComputePipelineState> p = m2Pipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeLinearM2(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), buffer(c.W), buffer(c.Y), c.batch, c.in, c.out,
                   16);
  });
}
#else
static std::string m2Problem() { return "lab 6.4's solution was not found when the lab was configured"; }
static pmpp::Timing runM2(Case&) { return {0, 0}; }
#endif

static std::string vendorProblem() { return ""; }   // MPS is part of macOS
static pmpp::Timing runVendor(Case& c, bool* ok) {
  *ok = true;
  return pmpp::run([&](void* v) {
    *ok = encodeLinearVendor((pmpp::metal::Pass*)v, buffer(c.X), buffer(c.W), buffer(c.Y), (uint32_t)c.batch,
                             (uint32_t)c.in, (uint32_t)c.out);
  });
}

static void kernelRow(const pmpp::Args& a, int width) {
  id<MTLComputePipelineState> p = m3Pipeline(width);
  if (!p) return;
  std::printf("pipeline of linear_m3 (SIMD_WIDTH=%d): threadExecutionWidth %lu, maxTotalThreadsPerThreadgroup %lu,\n"
              "  staticThreadgroupMemoryLength %lu bytes. Metal reports no registers per thread and has no\n"
              "  occupancy function (ch15-facts C3); Xcode's GPU tools show occupancy.\n",
              width, (unsigned long)p.threadExecutionWidth, (unsigned long)p.maxTotalThreadsPerThreadgroup,
              (unsigned long)p.staticThreadgroupMemoryLength);
  pmpp::bench_row(a, "pipeline").s("kernel", "m3").n("width", width)
      .n("threadExecutionWidth", (double)p.threadExecutionWidth)
      .n("maxTotalThreadsPerThreadgroup", (double)p.maxTotalThreadsPerThreadgroup)
      .n("staticThreadgroupMemoryLength", (double)p.staticThreadgroupMemoryLength).emit();
}
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }

#else  // CUDA, HIP and the emulators
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out);
void launchLinearWidth(const float* X, const float* W, float* Y, int batch, int in, int out, int width);
bool launchLinearVendor(const float* X, const float* W, float* Y, int batch, int in, int out);   // provided
int m3RegistersPerThread(int width);                                                              // provided
int m3BlocksPerSm(int width);                                                                     // provided
#if defined(PMPP_HAVE_M2)
void launchLinearM2(const float* X, const float* W, float* Y, int batch, int in, int out);       // lab 6.4's
#endif

static const float* dIn(Buf* b) { return (const float*)pmpp::raw(b); }
static float* dOut(Buf* b) { return (float*)pmpp::raw(b); }

static std::string m3Problem(int) {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (limit < 256) return "this device allows at most " + std::to_string(limit) + " threads per block";
  return "";
}
static pmpp::Timing runM3(Case& c, int width) {
  return pmpp::run([&](void*) { launchLinearWidth(dIn(c.X), dIn(c.W), dOut(c.Y), c.batch, c.in, c.out, width); });
}
static int deviceWidth() { return (int)pmpp::device_prop("warpSize", 32) == 64 ? 64 : 32; }
static std::string widthSource() {
  return "warpSize from the device properties: " + std::to_string((int)pmpp::device_prop("warpSize", 0));
}
static pmpp::Timing runM3Device(Case& c) {
  return pmpp::run([&](void*) { launchLinear(dIn(c.X), dIn(c.W), dOut(c.Y), c.batch, c.in, c.out); });
}

#if defined(PMPP_HAVE_M2)
static std::string m2Problem() { return ""; }
static pmpp::Timing runM2(Case& c) {
  return pmpp::run([&](void*) { launchLinearM2(dIn(c.X), dIn(c.W), dOut(c.Y), c.batch, c.in, c.out); });
}
#else
static std::string m2Problem() { return "lab 6.4's solution was not found when the lab was configured"; }
static pmpp::Timing runM2(Case&) { return {0, 0}; }
#endif

static std::string vendorProblem() {
  return std::string(vendorName()) == "none" ? "no vendor BLAS library was found when the lab was configured" : "";
}
static pmpp::Timing runVendor(Case& c, bool* ok) {
  *ok = true;
  return pmpp::run([&](void*) { *ok = launchLinearVendor(dIn(c.X), dIn(c.W), dOut(c.Y), c.batch, c.in, c.out); });
}

static void kernelRow(const pmpp::Args& a, int width) {
  const int regs = m3RegistersPerThread(width), blocks = m3BlocksPerSm(width);
  if (regs < 0 && blocks < 0) {
    std::printf("occupancy: the runtime was not asked (PMPP_CH15_OCCUPANCY_API is off, or the call failed)\n");
    return;
  }
  const double perSm = pmpp::device_prop("maxThreadsPerMultiProcessor", 0);
  const double occ = (blocks >= 0 && perSm > 0) ? blocks * 256.0 / perSm : -1;
  std::printf("occupancy of fastLinearKernel<%d>, as the runtime reports it: %d registers per thread, %d blocks of\n"
              "  256 threads per SM", width, regs, blocks);
  if (occ >= 0) std::printf(" = %.1f%% of maxThreadsPerMultiProcessor (%.0f)", 100 * occ, perSm);
  std::printf("\n");
  auto j = pmpp::bench_row(a, "occupancy");
  j.s("kernel", "m3").n("width", width).n("registers_per_thread", regs).n("blocks_per_sm", blocks);
  if (occ >= 0) j.n("occupancy", occ);
  j.emit();
}
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<Shape> shapes = lin::testShapes();
  // On a GPU each case runs twice, Y back to NaN in between: a value left over from the
  // first launch cannot pass the second.
  const int launches = pmpp::is_emu() ? 1 : 2;
  for (int width : {32, 64}) {
    const std::string who = "width " + std::to_string(width);
    const std::string problem = m3Problem(width);
    if (!problem.empty()) {
      t.check(who + ": the kernel runs here", [&] { return problem; });
      continue;
    }
    for (const Shape& s : shapes) {
      t.check(lin::caseName(who, s), [&]() -> std::string {
        Case c(s);
        std::unique_ptr<ch15::Watchdog> wd;
        if (s.in == 0) wd.reset(new ch15::Watchdog(a, lin::caseName(who, s), ch15::kEmptySumSeconds, lin::kEmptySumWhy));
        for (int l = 0; l < launches; ++l) {
          c.refill();
          runM3(c, width);
          const std::string err = c.check();
          if (!err.empty()) return launches > 1 ? "launch " + std::to_string(l + 1) + ": " + err : err;
        }
        return "";
      });
    }
  }

  // The way the project calls it: the width this device chose.
  {
    const Shape s{130, 9, 131, "partial tiles in every direction"};
    const std::string who = "as the project calls it, at this device's width " + std::to_string(deviceWidth());
    if (m3Problem(deviceWidth()).empty())
      t.check(lin::caseName(who, s), [&]() -> std::string {
        Case c(s);
        c.refill();
        runM3Device(c);
        return c.check();
      });
  }

  // The provided vendor call: checked against the same reference, so that the benchmark
  // compares like with like. A failure here is the course's bug, not yours.
  if (!pmpp::is_emu()) {
    const std::string vp = vendorProblem();
    if (!vp.empty()) {
      std::printf("  note: no vendor rows: %s\n", vp.c_str());
    } else {
      const std::vector<Shape> vshapes = {{1, 40, 260, "one token"},
                                          {17, 33, 9, "nothing is a multiple of anything"},
                                          {130, 9, 131, "partial tiles in every direction"},
                                          {257, 129, 511, "large and awkward"}};
      for (const Shape& s : vshapes) {
        t.check(lin::caseName(std::string("provided vendor call, ") + vendorName(), s), [&]() -> std::string {
          Case c(s);
          c.refill();
          bool ok = true;
          runVendor(c, &ok);
          if (!ok) return "the library refused the call";
          return c.check();
        });
      }
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
enum class K { M2, M3, Vendor };
static const char* kName(K k) { return k == K::M2 ? "m2" : k == K::M3 ? "m3" : "vendor"; }

static pmpp::Timing runK(K k, Case& c) {
  if (k == K::M2) return runM2(c);
  if (k == K::M3) return runM3Device(c);
  bool ok = true;
  pmpp::Timing t = runVendor(c, &ok);
  return ok ? t : pmpp::Timing{0, 0};
}

// Empty if kernel k gives the right Y on two awkward shapes (one per load and store path).
static std::string precheck(K k) {
  const Shape shapes[] = {{130, 9, 131, "scalar paths"}, {33, 72, 260, "float4 paths"}};
  for (const Shape& s : shapes) {
    Case c(s);
    c.refill();
    runK(k, c);
    const std::string err = c.check();
    if (!err.empty()) return "batch " + std::to_string(s.batch) + ", in " + std::to_string(s.in) + ", out " +
                             std::to_string(s.out) + ": " + err;
  }
  return "";
}

struct Layer { int in, out; std::vector<int> batches; const char* what; };

static int bench(const pmpp::Args& a) {
  const int width = deviceWidth();
  int rc = 0;
  const std::string m3p = m3Problem(width);
  if (!m3p.empty()) { std::printf("your kernel cannot run here: %s\n", m3p.c_str()); return 1; }

  std::printf("M3 runs at warp width %d (%s)\n", width, widthSource().c_str());
  kernelRow(a, width);

  // A timing of a wrong kernel means nothing: check each first.
  std::vector<K> ks;
  for (K k : {K::M2, K::M3, K::Vendor}) {
    std::string why = k == K::M2 ? m2Problem() : k == K::Vendor ? vendorProblem() : "";
    if (why.empty()) why = precheck(k);
    if (!why.empty()) {
      std::printf("%s not timed: %s\n", kName(k), why.c_str());
      if (k == K::M3) rc = 1;
      continue;
    }
    ks.push_back(k);
  }
  if (std::find(ks.begin(), ks.end(), K::M3) == ks.end()) return 1;
  const bool haveVendor = std::find(ks.begin(), ks.end(), K::Vendor) != ks.end();
  if (haveVendor) std::printf("vendor: %s; math mode: %s\n", vendorName(), vendorMath().c_str());
  if (std::find(ks.begin(), ks.end(), K::M2) != ks.end()) std::printf("m2: lab 6.4's kernel, T = 16\n");

  const Layer layers[] = {
      {1024, 4096, {1, 4, 16, 64, 256, 1024}, "the layer of labs 5.3 and 6.4"},
      {4096, 4096, {1, 16, 256}, "a 7B-class hidden size (our choice, not the book's)"},
  };
  const double budget = memoryBudget();
  std::printf("\nKernels take turns in every round; each figure is the best round. At least %d rounds,\n"
              "more for short kernels (up to 100, about 50 ms of the fastest kernel's time).\n", a.reps);
  for (const Layer& L : layers) {
    std::printf("\n%d -> %d: %s\n  %6s", L.in, L.out, L.what, "batch");
    for (K k : ks) std::printf("  %10s %9s", (std::string(kName(k)) + " ms").c_str(), "GFLOP/s");
    if (haveVendor) std::printf("  %12s", "m3 / vendor");
    std::printf("\n");
    for (int batch : L.batches) {
      const double bytes = 4.0 * ((double)batch * L.in + (double)L.out * L.in + (double)batch * L.out * 2 + 128.0 * L.out);
      if (budget > 0 && bytes > budget / 2) {
        std::printf("  %6d skipped: needs %.0f MB, more than half of what the device reports\n", batch, bytes / 1e6);
        continue;
      }
      Case c(batch, L.in, L.out);
      c.refill();
      // Warm-up, and a first time to size the number of rounds.
      double fastest = 1e30;
      for (K k : ks) fastest = std::min(fastest, std::max(runK(k, c).best(), 1e-7));
      const int rounds = std::max(a.reps, (int)std::min(100.0, std::ceil(0.05 / fastest)));
      std::vector<double> best(ks.size(), 1e30);
      for (int r = 0; r < rounds; ++r)
        for (size_t i = 0; i < ks.size(); ++i) best[i] = std::min(best[i], runK(ks[i], c).best());
      const std::string err = pmpp::take_error();
      if (!err.empty()) { std::printf("  %6d failed: %s\n", batch, err.c_str()); rc = 1; continue; }
      std::printf("  %6d", batch);
      double m3s = 0, vs = 0;
      for (size_t i = 0; i < ks.size(); ++i) {
        std::printf("  %10.4f %9.1f", best[i] * 1e3, c.flop() / best[i] / 1e9);
        if (ks[i] == K::M3) m3s = best[i];
        if (ks[i] == K::Vendor) vs = best[i];
      }
      if (haveVendor) std::printf("  %11.0f%%", 100.0 * vs / m3s);
      std::printf("\n");
      // Machine-readable rows go on lines of their own, after the table row.
      for (size_t i = 0; i < ks.size(); ++i) {
        auto j = pmpp::bench_row(a, kName(ks[i]));
        j.s("kernel", kName(ks[i])).n("batch", batch).n("in", L.in).n("out", L.out).n("seconds", best[i])
            .n("gflops", c.flop() / best[i] / 1e9).n("rounds", rounds);
        if (ks[i] == K::M2) j.n("tile", 16);
        if (ks[i] == K::M3) j.n("width", width);
        if (ks[i] == K::Vendor) j.s("library", vendorName()).s("math", vendorMath());
        j.emit();
      }
    }
  }
  std::printf("\nThese are measurements of this machine, not expectations.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
