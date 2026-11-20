// Lab 22.1 driver: tests two decompositions of a decode-time matrix-vector product (your
// gather kernel and the provided split-K kernel) against a CPU reference, then times both
// across layer shapes, batch sizes and split counts.
//   Y[batch x out] = X[batch x in] * W^T,  W stored (out, in) row-major, batch <= 8.
// You do not need to edit this file; your work goes in work/.
//
// What a test case does (README, "What the tests do"):
//   - X and W are allocated with exactly the floats they hold, so on `emu` a read past
//     either is an AddressSanitizer report naming the line.
//   - Y starts as NaN and is followed, in the same allocation, by a guard band of
//     sentinels that must survive.
//   - Each case launches twice without resetting Y in between, so a split-K launch that
//     does not zero Y itself fails the second time.
//   - Results are compared with y = x W^T computed in double, with a tolerance scaled by
//     sum_k |x w| for each output (as in lab 15.6). Never bitwise: atomics add in any order.
#include "pmpp/harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

using pmpp::Buf;

static const int kMaxBatch = 8;   // MAX_BATCH in the kernels

// ---------------------------------------------------------------- the CPU reference
// Y[b][o] = sum_k X[b][k] * W[o][k], accumulated in double; mag = sum_k |X[b][k] * W[o][k]|,
// which scales the tolerance.
static void cpuLinear(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
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
//   gatherProblem(), splitProblem()   empty if that kernel can run here, else why not
//   enqueueGather(pass, X, W, Y, ...)  one launch of your gather kernel
//   enqueueSplitK(pass, X, W, Y, ..., splits)  one launch of split-K (zeroing included)
//   warpWidth(), widthSource()         the warp (SIMD-group) width, and where it came from
//   residentWarps()                    what the device reports about resident warps
//   memoryBudget()                     bytes the benchmark may plan around
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeGemvGather(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                      id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvSplitK(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> splitPipeline,
                      id<MTLComputePipelineState> zeroPipeline, id<MTLBuffer> X, id<MTLBuffer> W, id<MTLBuffer> Y,
                      uint32_t batch, uint32_t in, uint32_t out, uint32_t splits);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
// Empty if the pipeline was built and can run at least one whole SIMD-group.
static std::string pipelineProblem(const char* fn) {
  id<MTLComputePipelineState> p = pso(fn);
  if (!p) return std::string(fn) + " did not build: " + pmpp::take_error();
  if (p.threadExecutionWidth == 0 || p.maxTotalThreadsPerThreadgroup < p.threadExecutionWidth)
    return std::string(fn) + ": this pipeline cannot run one whole SIMD-group per threadgroup";
  return "";
}
static std::string gatherProblem() { return pipelineProblem("gemv_gather"); }
static std::string splitProblem() {
  std::string e = pipelineProblem("gemv_splitk");
  if (e.empty()) e = pipelineProblem("gemv_zero");
  return e;
}
static void enqueueGather(void* vp, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  id<MTLComputePipelineState> p = pso("gemv_gather");
  if (!p) return;
  encodeGemvGather(((pmpp::metal::Pass*)vp)->encoder, p, buffer(X), buffer(W), buffer(Y), (uint32_t)batch,
                   (uint32_t)in, (uint32_t)out);
}
static void enqueueSplitK(void* vp, Buf* X, Buf* W, Buf* Y, int batch, int in, int out, int splits) {
  id<MTLComputePipelineState> p = pso("gemv_splitk"), z = pso("gemv_zero");
  if (!p || !z) return;
  encodeGemvSplitK(((pmpp::metal::Pass*)vp)->encoder, p, z, buffer(X), buffer(W), buffer(Y), (uint32_t)batch,
                   (uint32_t)in, (uint32_t)out, (uint32_t)splits);
}
// Apple documents no fixed SIMD-group width; the pipeline reports it (ch03-facts C9).
static int warpWidth() {
  id<MTLComputePipelineState> p = pso("gemv_gather");
  if (!p) p = pso("gemv_zero");
  return p ? (int)p.threadExecutionWidth : 0;
}
static std::string widthSource() { return "threadExecutionWidth of the gemv_gather pipeline"; }
static std::string residentWarps() {
  return "Metal reports no figure for how many SIMD-groups stay resident: there is no occupancy API\n"
         "  (Xcode's GPU tools show occupancy; ch06-facts C5, ch15-facts C3).";
}
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }

#else  // CUDA, HIP and the emulators
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out);
void launchGemvSplitK(const float* X, const float* W, float* Y, int batch, int in, int out, int splits);

static const float* dIn(Buf* b) { return (const float*)pmpp::raw(b); }
static float* dOut(Buf* b) { return (float*)pmpp::raw(b); }

// Blocks of 4 warps (WARPS_PER_BLOCK in kernel.cu).
static std::string blockProblem() {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  const int width = (int)pmpp::device_prop("warpSize", 0);
  if (width <= 0) return "the device reports no warpSize";
  if (limit < 4 * width) return "this device allows " + std::to_string(limit) + " threads per block; the kernels need 4 warps";
  return "";
}
static std::string gatherProblem() { return blockProblem(); }
static std::string splitProblem() { return blockProblem(); }
static void enqueueGather(void*, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  launchGemvGather(dIn(X), dIn(W), dOut(Y), batch, in, out);
}
static void enqueueSplitK(void*, Buf* X, Buf* W, Buf* Y, int batch, int in, int out, int splits) {
  launchGemvSplitK(dIn(X), dIn(W), dOut(Y), batch, in, out, splits);
}
static int warpWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static std::string widthSource() { return "warpSize from the device properties"; }
static std::string residentWarps() {
  const double sms = pmpp::device_prop("multiProcessorCount", 0);
  const double perSm = pmpp::device_prop("maxThreadsPerMultiProcessor", 0);
  const double width = pmpp::device_prop("warpSize", 0);
  if (sms <= 0 || perSm <= 0 || width <= 0) return "the device does not report how many warps it keeps resident";
  char b[300];
  std::snprintf(b, sizeof b,
                "the device reports %.0f multiprocessors x %.0f threads = at most %.0f warps resident at once",
                sms, perSm, sms * perSm / width);
  return b;
}
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

static pmpp::Timing runGather(Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  return pmpp::run([&](void* p) { enqueueGather(p, X, W, Y, batch, in, out); });
}
static pmpp::Timing runSplitK(Buf* X, Buf* W, Buf* Y, int batch, int in, int out, int splits) {
  return pmpp::run([&](void* p) { enqueueSplitK(p, X, W, Y, batch, in, out, splits); });
}

// ---------------------------------------------------------------- a test problem on the device
static const float kSentinel = -77777.0f;   // no correct kernel writes anything past Y

struct Shape { int batch, in, out; const char* why; };

struct Case {
  int batch, in, out;
  size_t guard;
  std::vector<float> x, w;
  const std::vector<float>* ref = nullptr;
  const std::vector<float>* mag = nullptr;
  Buf *X = nullptr, *W = nullptr, *Y = nullptr;

  Case(int b, int i, int o) : batch(b), in(i), out(o) {
    // On the emulators the band is short, so a write that runs further also leaves the
    // allocation (AddressSanitizer names the line on `emu`). On a device it is longer than
    // any row index below MAX_BATCH could reach.
    guard = pmpp::is_emu() ? 256 : (size_t)kMaxBatch * out + 256;
    const uint32_t seed = 22000u + (uint32_t)b * 131u + (uint32_t)i * 17u + (uint32_t)o;
    x = pmpp::random_floats((size_t)batch * in, seed);
    w = pmpp::random_floats((size_t)out * in, seed + 1);
    X = pmpp::alloc(x.size() * 4);
    W = pmpp::alloc(w.size() * 4);
    Y = pmpp::alloc(((size_t)batch * out + guard) * 4);
    pmpp::upload(X, x.data(), x.size() * 4);
    pmpp::upload(W, w.data(), w.size() * 4);
  }
  explicit Case(const Shape& s) : Case(s.batch, s.in, s.out) {}
  Case(const Case&) = delete;
  ~Case() { pmpp::release(X); pmpp::release(W); pmpp::release(Y); }

  // Y = NaN, guard band = sentinel.
  void refill() {
    std::vector<float> v((size_t)batch * out + guard, std::numeric_limits<float>::quiet_NaN());
    std::fill(v.begin() + (size_t)batch * out, v.end(), kSentinel);
    pmpp::upload(Y, v.data(), v.size() * 4);
  }

  // The reference depends only on the shape, so it is computed once per shape.
  void reference() {
    static std::map<std::tuple<int, int, int>, std::pair<std::vector<float>, std::vector<float>>> cache;
    auto& e = cache[std::make_tuple(batch, in, out)];
    if (e.first.empty()) cpuLinear(x, w, batch, in, out, e.first, e.second);
    ref = &e.first;
    mag = &e.second;
  }

  // Empty if Y matches the reference and the guard band is untouched.
  std::string check() {
    if (!ref) reference();
    const size_t n = (size_t)batch * out;
    std::vector<float> got(n + guard);
    pmpp::download(Y, got.data(), got.size() * 4);
    std::string err = pmpp::describe(pmpp::compare(got.data(), ref->data(), mag->data(), n), out);
    size_t changed = 0, first = 0;
    for (size_t i = n; i < got.size(); ++i) {
      if (std::memcmp(&got[i], &kSentinel, 4) != 0) {
        if (changed == 0) first = i - n;
        ++changed;
      }
    }
    if (changed) {
      char b[300];
      std::snprintf(b, sizeof b,
                    "%s%zu of the %zu floats after Y were overwritten; the first, %zu floats past the end, is "
                    "where row %zu, column %zu of Y would be if Y had more rows",
                    err.empty() ? "" : "; ", changed, guard, first, (size_t)batch + first / out, first % out);
      err += b;
    }
    return err;
  }
};

// "7 chunks of 5, the last 3" / "32 chunks of 2: 17 hold data, 15 are empty"
static std::string chunkText(int in, int splits) {
  if (splits == 1) return "one chunk: the whole row";
  const int chunk = (in + splits - 1) / splits;
  const int used = (in + chunk - 1) / chunk;
  const int last = in - (used - 1) * chunk;
  char b[160];
  if (used == splits)
    std::snprintf(b, sizeof b, "%d chunks of %d%s", splits, chunk,
                  last != chunk ? (", the last " + std::to_string(last)).c_str() : "");
  else
    std::snprintf(b, sizeof b, "%d chunks of %d: %d hold data, %d are empty", splits, chunk, used, splits - used);
  return b;
}

static std::string shapeText(const Shape& s) {
  char b[200];
  std::snprintf(b, sizeof b, "batch %d, in %d, out %d  (%s)", s.batch, s.in, s.out, s.why);
  return b;
}

// ---------------------------------------------------------------- tests
static std::vector<Shape> gatherShapes() {
  if (std::string(pmpp::backend()) == "race")
    return {{2, 64, 8, "two warp-widths of input"},
            {1, 31, 5, "in shorter than a warp"},
            {3, 100, 64, "in not a multiple of any warp width"}};
  std::vector<Shape> s = {
      {1, 1, 1, "one element"},
      {1, 31, 5, "in shorter than a warp"},
      {2, 33, 7, "in one past a 32-lane warp"},
      {3, 100, 64, "in not a multiple of any warp width"},
      {8, 257, 130, "the largest batch; nothing divides anything"},
      {5, 64, 1, "one output: one warp does all the work in gather"},
  };
  if (!pmpp::is_emu()) {
    s.push_back({1, 4096, 16, "a narrow layer such as a router; shape ours"});
    s.push_back({8, 4096, 4096, "a 4096-wide layer at the largest batch"});
    s.push_back({4, 14336, 4096, "an MLP down-projection; shape ours"});
    s.push_back({8, 1000, 4099, "out not a multiple of the 4 warps of a block"});
  }
  return s;
}

// 1, 2, 3, 7, 32 and one split per warp width of input. The emulators model 32-lane warps
// only, so there the split for a 64-lane warp is added by hand: it changes which lane
// reads what, and how many chunks are empty.
static std::vector<int> splitCounts(int in) {
  std::vector<int> s = {1, 2, 3, 7, 32};
  std::vector<int> widths;
  if (pmpp::is_emu()) widths = {32, 64};
  else if (warpWidth() > 0) widths = {warpWidth()};
  for (int w : widths) s.push_back(std::max(1, (in + w - 1) / w));
  std::sort(s.begin(), s.end());
  s.erase(std::unique(s.begin(), s.end()), s.end());
  return s;
}

// On the emulators every shuffle is a switch between threads, and a split-K launch has
// S x out warps. The chunking is along `in`, so the two widest shapes keep their batch and
// `in` there and lose most of their outputs.
static Shape splitShape(const Shape& s) {
  if (!pmpp::is_emu()) return s;
  if (s.batch == 8 && s.in == 257) return {8, 257, 5, "the largest batch; out cut to 5 on the emulators"};
  if (s.batch == 3 && s.in == 100) return {3, 100, 9, "in not a multiple of any warp width; out cut to 9 on the emulators"};
  return s;
}

struct SplitCase { Shape shape; int splits; };
static std::vector<SplitCase> splitCases() {
  std::vector<SplitCase> c;
  if (std::string(pmpp::backend()) == "race") {
    // S = 4 and 16 first: with 4 warps per block, every block then holds 4 warps that add
    // into the same outputs, and `race` runs the warps of a block at the same time.
    const Shape s = {2, 64, 8, "the race check: the warps that share an output share a block"};
    for (int S : {4, 16, 1, 2, 3, 7}) c.push_back({s, S});
    c.push_back({{3, 100, 9, "in not a multiple of any warp width"}, 7});
    return c;
  }
  for (const Shape& g : gatherShapes()) {
    const Shape s = splitShape(g);
    for (int S : splitCounts(s.in)) c.push_back({s, S});
  }
  return c;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("warp width: %d (%s)%s\n", warpWidth(), widthSource().c_str(),
              pmpp::is_emu() ? "; the emulators model 32-lane warps only" : "");

  // (a) your gather kernel
  for (const Shape& s : gatherShapes()) {
    t.check("gather: " + shapeText(s), [&]() -> std::string {
      const std::string why = gatherProblem();
      if (!why.empty()) return why;
      Case c(s);
      c.refill();
      for (int launch = 0; launch < 2; ++launch) {
        runGather(c.X, c.W, c.Y, s.batch, s.in, s.out);
        const std::string e = c.check();
        if (!e.empty()) return (launch ? "second launch on the same Y: " : "") + e;
      }
      return "";
    });
  }

  // (b) the provided split-K kernel
  for (const SplitCase& sc : splitCases()) {
    const Shape& s = sc.shape;
    t.check("split-K, S = " + std::to_string(sc.splits) + ", " + chunkText(s.in, sc.splits) + ": " + shapeText(s),
            [&]() -> std::string {
              const std::string why = splitProblem();
              if (!why.empty()) return why;
              Case c(s);
              c.refill();
              for (int launch = 0; launch < 2; ++launch) {
                runSplitK(c.X, c.W, c.Y, s.batch, s.in, s.out, sc.splits);
                const std::string e = c.check();
                if (!e.empty()) return (launch ? "second launch on the same Y: " : "") + e;
              }
              return "";
            });
  }

  // Real devices: the most contended case, 64 warps on each of 16 outputs, five launches.
  // A lost update shows as a wrong sum only when two of those warps happen to collide in
  // time, so this is evidence, not a guarantee (README).
  if (!pmpp::is_emu()) {
    t.check("split-K, contention: S = 64, five launches: batch 1, in 4096, out 16", [&]() -> std::string {
      const std::string why = splitProblem();
      if (!why.empty()) return why;
      Case c(1, 4096, 16);
      c.refill();
      int wrong = 0;
      std::string first;
      for (int launch = 0; launch < 5; ++launch) {
        runSplitK(c.X, c.W, c.Y, 1, 4096, 16, 64);
        const std::string e = c.check();
        if (!e.empty()) {
          if (wrong == 0) first = e;
          ++wrong;
        }
      }
      if (wrong) return "wrong in " + std::to_string(wrong) + " of 5 launches; the first: " + first;
      return "";
    });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// For each layer and batch, gather and split-K at S = 2, 4, 8, ... up to in / (8 x warp
// width) take turns, a.reps rounds after one untimed round, and each keeps its best round.
// One timed submission launches the kernel once per copy of W, going through copies that
// together hold at least 256 MB (at most 1024 copies), so that a launch does not find its
// weights where the previous launch left them. The time reported is per launch.
static int bench(const pmpp::Args& a) {
  const std::string gp = gatherProblem(), sp = splitProblem();
  if (!gp.empty() || !sp.empty()) {
    std::printf("bench: %s\n", (!gp.empty() ? gp : sp).c_str());
    return 1;
  }
  const int width = warpWidth();
  if (width <= 0) { std::printf("bench: the warp width is unknown\n"); return 1; }
  std::printf("warp width: %d (%s)\n", width, widthSource().c_str());
  std::printf("%s\n", residentWarps().c_str());

  struct Layer { int in, out; const char* what; };
  const Layer layers[] = {
      {4096, 4096, "4096 -> 4096"},
      {4096, 16, "4096 -> 16, a narrow layer such as a router (shape ours)"},
      {14336, 4096, "14336 -> 4096, an MLP down-projection (shape ours)"},
  };
  const int batches[] = {1, 4, 8};
  const double kRotateBytes = 256e6;
  const double budget = memoryBudget();

  for (const Layer& L : layers) {
    const double wBytes = 4.0 * L.in * L.out;
    int copies = (int)std::min(1024.0, std::ceil(kRotateBytes / wBytes));
    if (budget > 0) copies = (int)std::min<double>(copies, std::floor(budget / 4 / wBytes));
    if (copies < 1) {
      std::printf("\nlayer %s: skipped, one copy of W (%.0f MB) is more than a quarter of the %.0f MB the device "
                  "reports\n", L.what, wBytes / 1e6, budget / 1e6);
      continue;
    }
    const std::vector<float> hw = pmpp::random_floats((size_t)L.in * L.out, 2200u + (uint32_t)L.out);
    const std::vector<float> hx = pmpp::random_floats((size_t)kMaxBatch * L.in, 2201u);
    std::vector<Buf*> Ws(copies);
    for (Buf*& w : Ws) {
      w = pmpp::alloc(hw.size() * 4);
      pmpp::upload(w, hw.data(), hw.size() * 4);
    }
    Buf* X = pmpp::alloc(hx.size() * 4);
    pmpp::upload(X, hx.data(), hx.size() * 4);
    Buf* Y = pmpp::alloc((size_t)kMaxBatch * L.out * 4);

    std::vector<int> splits = {1};   // 1 = gather
    for (int s = 2; s <= L.in / (8 * width); s *= 2) splits.push_back(s);

    std::printf("\nlayer %s: W is %.1f MB; one submission = %d launches over %d copies of W (%.0f MB)\n", L.what,
                wBytes / 1e6, copies, copies, copies * wBytes / 1e6);
    for (int batch : batches) {
      std::vector<double> best(splits.size(), std::numeric_limits<double>::infinity());
      auto once = [&](size_t i) -> double {
        const int S = splits[i];
        const pmpp::Timing tm = pmpp::run([&](void* p) {
          for (Buf* w : Ws) {
            if (S == 1) enqueueGather(p, X, w, Y, batch, L.in, L.out);
            else enqueueSplitK(p, X, w, Y, batch, L.in, L.out, S);
          }
        });
        return tm.best() / copies;
      };
      for (size_t i = 0; i < splits.size(); ++i) once(i);   // untimed round
      for (int r = 0; r < a.reps; ++r)
        for (size_t i = 0; i < splits.size(); ++i) best[i] = std::min(best[i], once(i));
      const std::string err = pmpp::take_error();
      if (!err.empty()) {
        std::printf("  batch %d: a launch failed: %s\n", batch, err.c_str());
        continue;
      }

      std::printf("  batch %d   %-8s %5s %8s %12s %12s %10s\n", batch, "kernel", "S", "warps", "time (us)",
                  "W GB/s", "vs gather");
      size_t bi = 0;
      for (size_t i = 0; i < splits.size(); ++i) {
        const bool gather = splits[i] == 1;
        const double gbs = wBytes / best[i] / 1e9;
        std::printf("            %-8s %5d %8lld %12.2f %12.1f %9.2fx\n", gather ? "gather" : "split-K", splits[i],
                    (long long)splits[i] * L.out, best[i] * 1e6, gbs, best[0] / best[i]);
        pmpp::bench_row(a, "gemv").s("kernel", gather ? "gather" : "splitk").n("splits", splits[i]).n("batch", batch)
            .n("in", L.in).n("out", L.out).n("seconds", best[i]).n("weight_gb_per_s", gbs).emit();
        if (best[i] < best[bi]) bi = i;
      }
      if (bi == 0)
        std::printf("            best: gather\n");
      else
        std::printf("            best: split-K at S = %d, %.2fx the speed of gather (%.2f us against %.2f us)\n",
                    splits[bi], best[0] / best[bi], best[bi] * 1e6, best[0] * 1e6);
    }
    for (Buf* w : Ws) pmpp::release(w);
    pmpp::release(X);
    pmpp::release(Y);
  }
  std::printf("\nThese are measurements of this device at these shapes, not properties of the kernels.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
