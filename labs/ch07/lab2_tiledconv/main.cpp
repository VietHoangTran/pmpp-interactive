// Lab 7.2 driver: tests and benchmarks the two tiled convolution kernels.
// You do not need to edit this file; your work goes in work/.
//
// Kernels exercised
//   yours     tiled_halo    the block is the input tile, halo in shared memory
//                           (Fig. 7.12, part A)
//             tiled_cached  shared tile = output tile, halo taps from global memory
//                           (Fig. 7.15, part B)
//   provided  const         one thread per output, filter in constant memory, the same
//                           compile-time radius, 16 x 16 blocks in every build
//                           (Fig. 7.9), ../shared/conv_const.*
//
// Shapes are written W x H (width x height). Every test fills P with NaN first, so an
// output nobody writes fails; on a real device P is followed by a guard region that
// must come back untouched.
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <tuple>

#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

using pmpp::Buf;

enum class Kind { Halo, Cached, Const };
static const char* kindName(Kind k) {
  return k == Kind::Halo ? "tiled_halo" : k == Kind::Cached ? "tiled_cached" : "const";
}
static const int kConstBlock = 16;   // the provided baseline's block side, in every build
static const int kMaxRadius = 7;     // F_c holds 15 x 15 floats

// ---- entry points implemented in <variant>/ and ../shared/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeConvTiledHalo(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> N,
                         id<MTLBuffer> F, id<MTLBuffer> P, int32_t width, int32_t height, uint32_t tile,
                         uint32_t radius);
void encodeConvTiledCached(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> N,
                           id<MTLBuffer> F, id<MTLBuffer> P, int32_t width, int32_t height, uint32_t tile,
                           uint32_t radius);
void encodeConvConstBaseline(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> N,
                             id<MTLBuffer> F, id<MTLBuffer> P, int32_t width, int32_t height);
using pmpp::metal::buffer;

// The provided baseline's shader, ../shared/conv_const.metal (the build defines the directory).
static const std::string kSharedMetal = std::string(PMPP_METAL_SHARED_DIR) + "/conv_const.metal";
// One pipeline per kernel, tile width and radius: both size arrays or fix loop counts.
static id<MTLComputePipelineState> pso(Kind k, int tile, int r) {
  const std::string R = std::to_string(r);
  if (k == Kind::Const) return pmpp::metal::pipeline(kSharedMetal, "conv_const_base", {{"FILTER_RADIUS", R}});
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, k == Kind::Halo ? "conv_tiled_halo" : "conv_tiled_cached",
                               {{"TILE_WIDTH", std::to_string(tile)}, {"FILTER_RADIUS", R}});
}
static const char* kLimitName = "the pipeline's maxTotalThreadsPerThreadgroup";
// The most threads per group this kernel accepts here (0: it did not build).
static int maxThreads(Kind k, int tile, int r) {
  id<MTLComputePipelineState> p = pso(k, tile, r);
  return p ? (int)p.maxTotalThreadsPerThreadgroup : 0;
}
static pmpp::Timing launch(Kind k, int tile, int r, Buf* N, Buf* F, Buf* P, int w, int h,
                           const std::vector<float>& /*filter: already in F*/) {
  id<MTLComputePipelineState> p = pso(k, tile, r);
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    if (k == Kind::Halo) encodeConvTiledHalo(enc, p, buffer(N), buffer(F), buffer(P), w, h, tile, r);
    else if (k == Kind::Cached) encodeConvTiledCached(enc, p, buffer(N), buffer(F), buffer(P), w, h, tile, r);
    else encodeConvConstBaseline(enc, p, buffer(N), buffer(F), buffer(P), w, h);
  });
}
static std::vector<int> tilesToTest() { return {8, 16, 32}; }
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchConvTiledHalo(const float* N, const float* F_h, float* P, int r, int width, int height);
void launchConvTiledCached(const float* N, const float* F_h, float* P, int r, int width, int height);
void launchConvConstBaseline(const float* N, const float* F_h, float* P, int r, int width, int height);
static float* fp(Buf* b) { return (float*)pmpp::raw(b); }
static const char* kLimitName = "maxThreadsPerBlock";
static int maxThreads(Kind, int, int) { return (int)pmpp::device_prop("maxThreadsPerBlock", 1024); }
static pmpp::Timing launch(Kind k, int /*tile: fixed at compile time*/, int r, Buf* N, Buf* /*F*/, Buf* P, int w,
                           int h, const std::vector<float>& filter) {
  return pmpp::run([&](void*) {
    if (k == Kind::Halo) launchConvTiledHalo(fp(N), filter.data(), fp(P), r, w, h);
    else if (k == Kind::Cached) launchConvTiledCached(fp(N), filter.data(), fp(P), r, w, h);
    else launchConvConstBaseline(fp(N), filter.data(), fp(P), r, w, h);
  });
}
static std::vector<int> tilesToTest() { return {TILE_WIDTH}; }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

static int threadsNeeded(Kind k, int tile) { return k == Kind::Const ? kConstBlock * kConstBlock : tile * tile; }
static int outTile(Kind k, int tile, int r) {
  return k == Kind::Halo ? tile - 2 * r : k == Kind::Cached ? tile : kConstBlock;
}

// ---- the CPU reference: zero ghost cells, no flip, accumulated in double ----
// `mag` receives sum |f * x| per output, which scales the tolerance (Metal does not
// default to strict IEEE arithmetic).
static void cpuConv(const std::vector<float>& n, const std::vector<float>& f, int w, int h, int r,
                    std::vector<float>& ref, std::vector<float>& mag) {
  const int m = 2 * r + 1;
  ref.assign((size_t)w * h, 0.f);
  mag.assign((size_t)w * h, 0.f);
  for (int row = 0; row < h; ++row)
    for (int col = 0; col < w; ++col) {
      double acc = 0, am = 0;
      for (int fRow = 0; fRow < m; ++fRow) {
        const int inRow = row - r + fRow;
        if (inRow < 0 || inRow >= h) continue;
        for (int fCol = 0; fCol < m; ++fCol) {
          const int inCol = col - r + fCol;
          if (inCol < 0 || inCol >= w) continue;
          const double p = (double)f[fRow * m + fCol] * n[(size_t)inRow * w + inCol];
          acc += p;
          am += std::fabs(p);
        }
      }
      ref[(size_t)row * w + col] = (float)acc;
      mag[(size_t)row * w + col] = (float)am;
    }
}

// The book's example (Fig. 7.4, p. 163): a 7 x 7 input and a 5 x 5 filter.
static const float kBookN[49] = {1, 2, 3, 4, 5, 6, 7,  2, 3, 4, 5, 6, 7, 8,  3, 4, 5, 6, 7, 8, 9,
                                 4, 5, 6, 7, 8, 5, 6,  5, 6, 7, 8, 5, 6, 7,  6, 7, 8, 9, 0, 1, 2,
                                 7, 8, 9, 0, 1, 2, 3};
static const float kBookF[25] = {1, 2, 3, 2, 1,  2, 3, 4, 3, 2,  3, 4, 5, 4, 3,  2, 3, 4, 3, 2,  1, 2, 3, 2, 1};

// ---- one convolution problem on the device ----
static const float kSentinel = -12345.6789f;
struct Conv {
  int w, h, r = 0;
  size_t guard;                 // floats after P that must stay untouched (0 on the emulators,
                                // where AddressSanitizer checks the exact allocation instead)
  std::vector<float> n, f;      // host copies of the input and the current filter
  Buf *N = nullptr, *F = nullptr, *P = nullptr;
  Conv(int w_, int h_, std::vector<float> n_) : w(w_), h(h_), guard(pmpp::is_emu() ? 0 : 65536), n(std::move(n_)) {
    N = pmpp::alloc(n.size() * 4);
    F = pmpp::alloc((size_t)(2 * kMaxRadius + 1) * (2 * kMaxRadius + 1) * 4);
    P = pmpp::alloc(((size_t)w * h + guard) * 4);
    pmpp::upload(N, n.data(), n.size() * 4);
  }
  Conv(const Conv&) = delete;
  ~Conv() { pmpp::release(N); pmpp::release(F); pmpp::release(P); }
  void setFilter(int r_, std::vector<float> f_) {
    r = r_;
    f = std::move(f_);
    pmpp::upload(F, f.data(), f.size() * 4);   // Metal reads this buffer; CUDA and HIP copy f to F_c
  }
  void fillP() {
    std::vector<float> v((size_t)w * h + guard, std::numeric_limits<float>::quiet_NaN());
    for (size_t i = (size_t)w * h; i < v.size(); ++i) v[i] = kSentinel;
    pmpp::upload(P, v.data(), v.size() * 4);
  }
  pmpp::Timing run(Kind k, int tile) { return launch(k, tile, r, N, F, P, w, h, f); }
  // Runs kernel `k` once on a NaN-filled P and compares with the reference. "" = correct.
  std::string runAndCheck(Kind k, int tile, const std::vector<float>& ref, const std::vector<float>& mag) {
    fillP();
    run(k, tile);
    const std::string launchErr = pmpp::take_error();   // a Metal compile error lands here too
    if (!launchErr.empty()) return "launch error: " + launchErr;
    std::vector<float> got((size_t)w * h + guard);
    pmpp::download(P, got.data(), got.size() * 4);
    for (size_t i = (size_t)w * h; i < got.size(); ++i)
      if (std::memcmp(&got[i], &kSentinel, 4) != 0)
        return "the kernel wrote past the end of P (" + std::to_string(i - (size_t)w * h) +
               " floats after the last output)";
    pmpp::Cmp c = pmpp::compare(got.data(), ref.data(), mag.data(), (size_t)w * h);
    if (c.ok) return "";
    size_t unwritten = 0;
    for (size_t i = 0; i < (size_t)w * h; ++i) unwritten += std::isnan(got[i]) ? 1 : 0;
    std::string d = pmpp::describe(c, w);
    if (unwritten) d += "; " + std::to_string(unwritten) + " of them never written (still NaN)";
    return d;
  }
};

// Test data, identical on every platform. The filter is random, hence asymmetric, so a
// flipped or transposed filter index fails.
static std::vector<float> inputFor(int w, int h) { return pmpp::random_floats((size_t)w * h, 17 + 31 * w + h); }
static std::vector<float> filterFor(int r) { return pmpp::random_floats((size_t)(2 * r + 1) * (2 * r + 1), 500 + r); }

struct Ref { std::vector<float> ref, mag; };
static const Ref& referenceFor(int w, int h, int r, const std::vector<float>& n, const std::vector<float>& f) {
  static std::map<std::tuple<int, int, int>, Ref> cache;   // shared by both kernels and every tile
  auto key = std::make_tuple(w, h, r);
  auto it = cache.find(key);
  if (it == cache.end()) {
    Ref x;
    cpuConv(n, f, w, h, r, x.ref, x.mag);
    it = cache.emplace(key, std::move(x)).first;
  }
  return it->second;
}

static std::string shape(int w, int h) { return std::to_string(w) + "x" + std::to_string(h); }

// "" if kernel k can run at this tile width and radius here; else the limit it exceeds.
static std::string limitProblem(Kind k, int tile, int r) {
  const int lim = maxThreads(k, tile, r);
  if (lim == 0) return "";   // it did not build: let the test run and report the compile error
  if (threadsNeeded(k, tile) <= lim) return "";
  return "needs " + std::to_string(threadsNeeded(k, tile)) + " threads per group; " + kLimitName + " is " +
         std::to_string(lim);
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const bool emu = pmpp::is_emu();
  const std::vector<std::pair<int, int>> shapes =
      emu ? std::vector<std::pair<int, int>>{{1, 1}, {7, 7}, {13, 9}, {33, 17}, {40, 31}}
          : std::vector<std::pair<int, int>>{{1, 1}, {7, 7}, {30, 30}, {33, 17}, {100, 61}, {257, 129}, {1000, 999}};
  const int maxR = emu ? 3 : kMaxRadius;

  // The provided baseline the benchmark compares against.
  for (int r = 1; r <= maxR; ++r) {
    t.check("provided const kernel, r = " + std::to_string(r) + ", 33x17", [&]() -> std::string {
      Conv c(33, 17, inputFor(33, 17));
      c.setFilter(r, filterFor(r));
      const Ref& x = referenceFor(33, 17, r, c.n, c.f);
      return c.runAndCheck(Kind::Const, kConstBlock, x.ref, x.mag);
    });
  }

  for (int tile : tilesToTest()) {
    const int rmax = std::min(maxR, (tile - 2) / 2);   // the halo kernel's output tile stays >= 2
#if !defined(PMPP_BACKEND_METAL)
    // This build has one tile width; a device that cannot run it is reported as lab 5.2
    // does. (HIP checks the block size at launch, notes/today/ch07-facts.md B6.) On Metal
    // the limit belongs to each pipeline, and is checked per kernel and radius below.
    if (!limitProblem(Kind::Halo, tile, 1).empty()) {
      t.check("tile " + std::to_string(tile) + " usable on this device",
              [&] { return limitProblem(Kind::Halo, tile, 1) + "; try the t16 or t8 build"; });
      continue;
    }
#endif
    for (Kind k : {Kind::Halo, Kind::Cached}) {
      const std::string head = "tile " + std::to_string(tile) + ", " + kindName(k) + ", ";
      // 1. The book's example: P[2][2] = 321 and P[1][0] = 112 (pp. 163-164).
      if (rmax >= 2) {
        std::string lim = limitProblem(k, tile, 2);
        if (!lim.empty()) std::printf("  %sr = 2 skipped: %s\n", head.c_str(), lim.c_str());
        else
          t.check(head + "the book's 7x7 example, r = 2", [&]() -> std::string {
            Conv c(7, 7, std::vector<float>(kBookN, kBookN + 49));
            c.setFilter(2, std::vector<float>(kBookF, kBookF + 25));
            std::vector<float> ref, mag;
            cpuConv(c.n, c.f, 7, 7, 2, ref, mag);
            std::string err = c.runAndCheck(k, tile, ref, mag);
            if (!err.empty()) return err;
            std::vector<float> got(49);
            pmpp::download(c.P, got.data(), 49 * 4);
            if (got[2 * 7 + 2] != 321.f || got[1 * 7 + 0] != 112.f) {
              char b[120];
              std::snprintf(b, sizeof b, "P[2][2] = %g and P[1][0] = %g; the book has 321 and 112", got[16], got[7]);
              return b;
            }
            return "";
          });
      }
      // 2. Random data, every radius the output tile allows, awkward shapes.
      for (int r = 1; r <= rmax; ++r) {
        std::string lim = limitProblem(k, tile, r);
        if (!lim.empty()) {
          std::printf("  %sr = %d skipped: %s\n", head.c_str(), r, lim.c_str());
          continue;
        }
        for (auto [w, h] : shapes) {
          t.check(head + "r = " + std::to_string(r) + ", " + shape(w, h), [&, w = w, h = h]() -> std::string {
            Conv c(w, h, inputFor(w, h));
            c.setFilter(r, filterFor(r));
            const Ref& x = referenceFor(w, h, r, c.n, c.f);
            return c.runAndCheck(k, tile, x.ref, x.mag);
          });
        }
      }
    }
  }
  return t.finish();
}

// ---- benchmark (real devices only) ----

// Best time of at least `reps` runs, repeated until the runs add up to a quarter of a
// second (short kernels vary a lot from run to run). 0 if a run failed; the reason
// goes to `why`.
static double bestTime(const pmpp::Args& a, Conv& c, Kind k, int tile, std::string* why = nullptr) {
  pmpp::Stat s = pmpp::time_it(1, [&] { return c.run(k, tile); });   // warm-up, then one run
  double best = s.min_s, total = s.min_s;
  for (int i = 1; i < a.reps || (total < 0.25 && i < 200); ++i) {
    double one = c.run(k, tile).best();
    best = std::min(best, one);
    total += one;
  }
  const std::string err = pmpp::take_error();
  if (why) *why = err;
  return err.empty() ? best : 0;
}

// Section 7.5's arithmetic intensity of the halo kernel, for an internal tile
// (book p. 176): (m^2/4) / (1/2 (1 + (m-1)/t)^2 + 1/2), with t the output tile width.
static double modelIntensity(int m, int t) {
  const double q = 1.0 + (m - 1.0) / t;
  return (m * m / 4.0) / (0.5 * q * q + 0.5);
}

static int bench(const pmpp::Args& a) {
  // Size the image as lab 7.1 does: start at 2048 x 2048 and double the side until the
  // baseline (const, r = 2) takes over 50 ms, at most 8192 x 8192, and keep N and P
  // within an eighth of the memory the device reports (and 2 GiB).
  double cap = 2.0 * 1024 * 1024 * 1024;
  const double mem = deviceMemory();
  if (mem > 0) cap = std::min(cap, mem / 8.0);
  int side = 2048;
  std::unique_ptr<Conv> c;
  for (;;) {
    c.reset(new Conv(side, side, pmpp::random_floats((size_t)side * side, 99)));
    c->setFilter(2, filterFor(2));
    std::string why;
    const double s = bestTime(a, *c, Kind::Const, kConstBlock, &why);
    if (s <= 0) {
      std::printf("the provided const kernel failed to run: %s\n", why.c_str());
      return 1;
    }
    const double next = 8.0 * (2.0 * side) * (2.0 * side);
    if (s > 0.05 || side >= 8192 || next > cap) break;
    c.reset();
    side *= 2;
  }
  const int w = side, h = side;
  std::printf("image %d x %d floats; each time is the best of at least %d runs\n", w, h, a.reps);
  std::printf("GFLOP/s counts 2 * m^2 per output, ghost taps included (the book's count, p. 167).\n");
  std::printf("model: section 7.5's FLOP/B for the halo kernel at t = out_tile, and that over the\n"
              "       const kernel's 0.5 FLOP/B (how many times less global traffic it asks for)\n\n");
  std::printf("  %-13s %5s %4s %3s %3s %10s %10s %8s %8s %8s\n", "kernel", "tile", "out", "r", "m", "ms", "GFLOP/s",
              "x const", "model", "ratio");

  std::vector<int> tiles;
#if defined(PMPP_BACKEND_METAL)
  tiles = {8, 16, 32};
#else
  tiles = {TILE_WIDTH};
#endif
  for (int r : {1, 2, 3, 5, 7}) {
    const int m = 2 * r + 1;
    const double flop = 2.0 * w * (double)h * m * m;
    c->setFilter(r, filterFor(r));

    // A small problem to check each kernel before it is timed.
    const int cw = 100, ch = 61;
    Conv small(cw, ch, inputFor(cw, ch));
    small.setFilter(r, filterFor(r));
    const Ref& x = referenceFor(cw, ch, r, small.n, small.f);

    std::string why = small.runAndCheck(Kind::Const, kConstBlock, x.ref, x.mag);
    if (!why.empty()) {
      std::printf("  const at r = %d gave a wrong result on %dx%d, nothing timed: %s\n", r, cw, ch, why.c_str());
      continue;
    }
    const double base = bestTime(a, *c, Kind::Const, kConstBlock, &why);
    if (base <= 0) {
      std::printf("  const at r = %d failed to run: %s\n", r, why.c_str());
      continue;
    }
    std::printf("  %-13s %5d %4d %3d %3d %10.3f %10.1f %8.2f\n", "const", kConstBlock, kConstBlock, r, m, base * 1e3,
                flop / base / 1e9, 1.0);
    pmpp::bench_row(a, "const").n("tile", kConstBlock).n("out_tile", kConstBlock).n("radius", r).n("m", m)
        .n("width", w).n("height", h).n("seconds", base).n("gflops", flop / base / 1e9).n("speedup_vs_const", 1.0)
        .emit();

    for (int tile : tiles) {
      if (tile - 2 * r < 2) continue;   // the halo kernel's output tile would be under 2
      for (Kind k : {Kind::Halo, Kind::Cached}) {
        const int ot = outTile(k, tile, r);
        std::string lim = limitProblem(k, tile, r);
        if (!lim.empty()) {
          std::printf("  %-13s %5d %4d %3d %3d   skipped: %s\n", kindName(k), tile, ot, r, m, lim.c_str());
          continue;
        }
        std::string err = small.runAndCheck(k, tile, x.ref, x.mag);
        if (!err.empty()) {
          std::printf("  %-13s %5d %4d %3d %3d   wrong result on %dx%d, not timed: %s\n", kindName(k), tile, ot, r, m,
                      cw, ch, err.c_str());
          continue;
        }
        const double s = bestTime(a, *c, k, tile, &why);
        if (s <= 0) {
          std::printf("  %-13s %5d %4d %3d %3d   failed to run: %s\n", kindName(k), tile, ot, r, m, why.c_str());
          continue;
        }
        pmpp::Json row = pmpp::bench_row(a, kindName(k));
        row.n("tile", tile).n("out_tile", ot).n("radius", r).n("m", m).n("width", w).n("height", h).n("seconds", s)
            .n("gflops", flop / s / 1e9).n("speedup_vs_const", base / s);
        if (k == Kind::Halo) {
          const double mi = modelIntensity(m, ot);
          std::printf("  %-13s %5d %4d %3d %3d %10.3f %10.1f %8.2f %8.2f %8.2f\n", kindName(k), tile, ot, r, m,
                      s * 1e3, flop / s / 1e9, base / s, mi, mi / 0.5);
          row.n("model_intensity", mi).n("model_traffic_ratio", mi / 0.5);
        } else {
          std::printf("  %-13s %5d %4d %3d %3d %10.3f %10.1f %8.2f\n", kindName(k), tile, ot, r, m, s * 1e3,
                      flop / s / 1e9, base / s);
        }
        row.emit();
      }
    }
  }
  std::printf("\nThese are measurements of this device. 'x const' is the const kernel's time divided by\n"
              "the kernel's; 'ratio' is what the model says about requested bytes, not about time.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
