// Lab 7.1 driver: tests and benchmarks your three 2D convolution kernels. They compute
// the same thing and differ only in where the filter lives: in global memory, in
// constant memory, or in a struct passed by value with the launch.
// You do not need to edit this file; your work goes in work/.
#include <limits>
#include <memory>

#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
#define MAX_FILTER_RADIUS 7   // the same limit as in kernel.cu and kernel.metal

using pmpp::Buf;

enum Kind { kBasic = 0, kConst = 1, kByValue = 2 };
static const Kind kKinds[3] = {kBasic, kConst, kByValue};
static const char* kNames[3] = {"basic", "const", "byvalue"};
// The book's model (pp. 167 and 171): 2 FLOP per 8 bytes of global memory with the
// filter in global memory; 2 FLOP per 4 bytes once no filter read reaches DRAM. The
// by-value filter travels in constant memory on CUDA (book p. 172), so it gets 0.5 too.
static const double kModelIntensity[3] = {0.25, 0.5, 0.5};

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeConvBasic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> N, id<MTLBuffer> F,
                     id<MTLBuffer> P, int32_t r, int32_t width, int32_t height, uint32_t tileWidth);
void encodeConvConst(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> N, id<MTLBuffer> F,
                     id<MTLBuffer> P, int32_t r, int32_t width, int32_t height, uint32_t tileWidth);
void encodeConvByValue(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> N, const float* F_h,
                       id<MTLBuffer> P, int32_t r, int32_t width, int32_t height, uint32_t tileWidth);
using pmpp::metal::buffer;

static const char* kFunctions[3] = {"conv_basic", "conv_const", "conv_by_value"};
static id<MTLComputePipelineState> pso(Kind k) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, kFunctions[k]); }
// Empty if kernel k can run here, else the reason it cannot.
static std::string unusable(Kind k) {
  id<MTLComputePipelineState> p = pso(k);
  if (!p) return std::string("kernel ") + kFunctions[k] + " did not build or is missing: " + pmpp::take_error();
  if ((NSUInteger)(TILE_WIDTH * TILE_WIDTH) > p.maxTotalThreadsPerThreadgroup)
    return std::string("a ") + std::to_string(TILE_WIDTH) + " x " + std::to_string(TILE_WIDTH) +
           " threadgroup exceeds this pipeline's maxTotalThreadsPerThreadgroup (" +
           std::to_string((unsigned long)p.maxTotalThreadsPerThreadgroup) + ")";
  return "";
}
static pmpp::Timing runConv(Kind k, Buf* N, Buf* F, const float* F_h, Buf* P, int r, int w, int h) {
  id<MTLComputePipelineState> p = pso(k);
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (k) {
      case kBasic:   encodeConvBasic(enc, p, buffer(N), buffer(F), buffer(P), r, w, h, TILE_WIDTH); break;
      case kConst:   encodeConvConst(enc, p, buffer(N), buffer(F), buffer(P), r, w, h, TILE_WIDTH); break;
      case kByValue: encodeConvByValue(enc, p, buffer(N), F_h, buffer(P), r, w, h, TILE_WIDTH); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchConvBasic(const float* N, const float* F, float* P, int r, int width, int height);
void launchConvConst(const float* N, const float* F_h, float* P, int r, int width, int height);
void launchConvByValue(const float* N, const float* F_h, float* P, int r, int width, int height);
static float* fp(Buf* b) { return (float*)pmpp::raw(b); }
static std::string unusable(Kind) {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (TILE_WIDTH * TILE_WIDTH > limit)
    return "a " + std::to_string(TILE_WIDTH) + " x " + std::to_string(TILE_WIDTH) +
           " block exceeds this device's maxThreadsPerBlock (" + std::to_string(limit) + ")";
  return "";
}
// F is the filter in device memory (for the basic kernel); F_h the same filter in
// host memory (for the two kernels whose launch copies it).
static pmpp::Timing runConv(Kind k, Buf* N, Buf* F, const float* F_h, Buf* P, int r, int w, int h) {
  return pmpp::run([&](void*) {
    switch (k) {
      case kBasic:   launchConvBasic(fp(N), fp(F), fp(P), r, w, h); break;
      case kConst:   launchConvConst(fp(N), F_h, fp(P), r, w, h); break;
      case kByValue: launchConvByValue(fp(N), F_h, fp(P), r, w, h); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

static const float kNaN = std::numeric_limits<float>::quiet_NaN();

// The CPU reference: the book's formula (p. 162), zero ghost cells, filter not
// flipped, accumulated in double. `mag` receives sum |f*x| for each output, which
// scales the tolerance (Metal does not default to strict IEEE arithmetic).
static void cpuConv(const std::vector<float>& N, const std::vector<float>& F, int r, int w, int h,
                    std::vector<float>& P, std::vector<float>& mag) {
  const int m = 2 * r + 1;
  P.assign((size_t)w * h, 0.f);
  mag.assign((size_t)w * h, 0.f);
  for (int row = 0; row < h; ++row)
    for (int col = 0; col < w; ++col) {
      double acc = 0, am = 0;
      for (int fRow = 0; fRow < m; ++fRow)
        for (int fCol = 0; fCol < m; ++fCol) {
          const int inRow = row - r + fRow, inCol = col - r + fCol;
          if (inRow < 0 || inRow >= h || inCol < 0 || inCol >= w) continue;
          const double p = (double)F[(size_t)fRow * m + fCol] * N[(size_t)inRow * w + inCol];
          acc += p;
          am += std::fabs(p);
        }
      P[(size_t)row * w + col] = (float)acc;
      mag[(size_t)row * w + col] = (float)am;
    }
}

// One problem on the device: image N (h x w), filter F ((2r+1) x (2r+1)), output P.
// The filter is kept on the host too (n, f) for the launches that copy it.
struct ConvProblem {
  int w, h, r;
  std::vector<float> n, f;
  Buf *N = nullptr, *F = nullptr, *P = nullptr;
  ConvProblem(int width, int height, int radius, std::vector<float> image, std::vector<float> filter,
              bool poisonOutput = true)
      : w(width), h(height), r(radius), n(std::move(image)), f(std::move(filter)) {
    const size_t e = (size_t)w * h;
    N = pmpp::alloc(e * 4); P = pmpp::alloc(e * 4); F = pmpp::alloc(f.size() * 4);
    pmpp::upload(N, n.data(), e * 4);
    pmpp::upload(F, f.data(), f.size() * 4);
    if (poisonOutput) {   // an output that is never written stays NaN and is reported
      std::vector<float> poison(e, kNaN);
      pmpp::upload(P, poison.data(), e * 4);
    }
  }
  ConvProblem(const ConvProblem&) = delete;
  ~ConvProblem() { pmpp::release(N); pmpp::release(F); pmpp::release(P); }
  pmpp::Timing run(Kind k) { return runConv(k, N, F, f.data(), P, r, w, h); }
  std::vector<float> result() const {
    std::vector<float> got((size_t)w * h);
    pmpp::download(P, got.data(), got.size() * 4);
    return got;
  }
  // Empty if `got` matches the CPU reference.
  std::string check(const std::vector<float>& got) const {
    std::vector<float> ref, mag;
    cpuConv(n, f, r, w, h, ref, mag);
    pmpp::Cmp c = pmpp::compare(got.data(), ref.data(), mag.data(), got.size());
    if (c.ok) return "";
    const size_t i = c.first_bad;
    char b[260];
    std::snprintf(b, sizeof b, "%zu of %zu outputs wrong; first at P[%zu][%zu]: got %g, expected %g",
                  c.bad, got.size(), i / (size_t)w, i % (size_t)w, (double)got[i], (double)ref[i]);
    return b;
  }
};

static std::string num(double v) { char b[40]; std::snprintf(b, sizeof b, "%g", v); return b; }

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const bool emu = pmpp::is_emu();

  // Fig. 7.4 (p. 163): the book's 7 x 7 input and 5 x 5 filter.
  const std::vector<float> bookN = {1, 2, 3, 4, 5, 6, 7,  2, 3, 4, 5, 6, 7, 8,  3, 4, 5, 6, 7, 8, 9,
                                    4, 5, 6, 7, 8, 5, 6,  5, 6, 7, 8, 5, 6, 7,  6, 7, 8, 9, 0, 1, 2,
                                    7, 8, 9, 0, 1, 2, 3};
  const std::vector<float> bookF = {1, 2, 3, 2, 1,  2, 3, 4, 3, 2,  3, 4, 5, 4, 3,  2, 3, 4, 3, 2,  1, 2, 3, 2, 1};

  // Random images with random (so asymmetric) filters: a flipped filter or a
  // transposed filter index fails. Sizes are width x height; 1 x 1 and the radius-7
  // cases on small images are almost all ghost taps.
  const std::vector<std::pair<int, int>> sizes =
      emu ? std::vector<std::pair<int, int>>{{1, 1}, {5, 9}, {7, 7}, {33, 17}}
          : std::vector<std::pair<int, int>>{{1, 1}, {5, 9}, {16, 16}, {33, 17}, {100, 61}, {257, 129}};
  const int radii[] = {0, 1, 2, 3, MAX_FILTER_RADIUS};

  for (Kind k : kKinds) {
    const std::string name = kNames[k];
    t.check(name + ": the book's example, 7 x 7 image, 5 x 5 filter (Fig. 7.4)", [&]() -> std::string {
      std::string why = unusable(k);
      if (!why.empty()) return why;
      ConvProblem p(7, 7, 2, bookN, bookF);
      p.run(k);
      std::vector<float> got = p.result();
      // The two outputs the book works out (pp. 163 and 164). Small integers, so exact.
      if (!(got[2 * 7 + 2] == 321.f)) return "P[2][2] is " + num(got[2 * 7 + 2]) + "; the book's answer is 321";
      if (!(got[1 * 7 + 0] == 112.f))
        return "P[1][0] is " + num(got[1 * 7 + 0]) + "; the book's answer is 112 (its patch has two ghost columns and one ghost row)";
      return p.check(got);
    });
    for (const auto& size : sizes) {
      const int w = size.first, h = size.second;
      t.check(name + ": " + std::to_string(w) + " x " + std::to_string(h) + " image, radius 0, 1, 2, 3 and 7",
              [&]() -> std::string {
        std::string why = unusable(k);
        if (!why.empty()) return why;
        for (int r : radii) {
          const int m = 2 * r + 1;
          const uint32_t seed = 1000u + 31u * (uint32_t)w + 7u * (uint32_t)h + (uint32_t)r;
          ConvProblem p(w, h, r, pmpp::random_floats((size_t)w * h, seed), pmpp::random_floats((size_t)m * m, seed + 500));
          p.run(k);
          std::string err = pmpp::take_error();
          if (!err.empty()) return "radius " + std::to_string(r) + ": launch error: " + err;
          err = p.check(p.result());
          if (!err.empty()) return "radius " + std::to_string(r) + ": " + err;
        }
        return "";
      });
    }
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // The image grows from 2048 x 2048 until the basic kernel takes over 50 ms, up to
  // 8192 x 8192, keeping N and P within a quarter of the memory the device reports.
  const double mem = deviceMemory();
  int cap = 8192;
  while (cap > 2048 && mem > 0 && 8.0 * cap * cap > mem / 4) cap /= 2;

#if defined(PMPP_BACKEND_METAL)
  if (id<MTLComputePipelineState> p = pso(kConst)) {
    std::printf("conv_const pipeline: threadExecutionWidth %lu, maxTotalThreadsPerThreadgroup %lu\n",
                (unsigned long)p.threadExecutionWidth, (unsigned long)p.maxTotalThreadsPerThreadgroup);
    pmpp::bench_row(a, "pipeline").s("kernel", "conv_const")
        .n("threadExecutionWidth", (double)p.threadExecutionWidth)
        .n("maxTotalThreadsPerThreadgroup", (double)p.maxTotalThreadsPerThreadgroup).emit();
  }
#endif
  std::printf("%d x %d threads per block, one thread per output, best of %d runs.\n", TILE_WIDTH, TILE_WIDTH, a.reps);
  std::printf("GFLOP/s counts 2 x width x height x m^2 per run, ghost taps included (the book's count, p. 167).\n");

  const int radii[] = {1, 2, 3, 5, 7};
  for (int r : radii) {
    const int m = 2 * r + 1;
    const std::vector<float> filter = pmpp::random_floats((size_t)m * m, 70 + r);
    std::unique_ptr<ConvProblem> p;
    int side = 2048;
    double basic_s = 0;
    for (;;) {
      p.reset();   // free the previous image before allocating the next
      p.reset(new ConvProblem(side, side, r, pmpp::random_floats((size_t)side * side, 9), filter, false));
      basic_s = pmpp::time_it(a.reps, [&] { return p->run(kBasic); }).min_s;
      if (basic_s <= 0 || basic_s > 0.05 || side >= cap) break;
      side *= 2;
    }
    const double flop = 2.0 * side * (double)side * m * m;
    std::printf("\nradius %d (%d x %d filter), image %d x %d, %.2f GFLOP per run\n", r, m, m, side, side, flop / 1e9);
    for (Kind k : kKinds) {
      const double s = k == kBasic ? basic_s : pmpp::time_it(a.reps, [&] { return p->run(k); }).min_s;
      const std::string err = pmpp::take_error();
      if (!err.empty() || s <= 0) { std::printf("  %-8s failed: %s\n", kNames[k], err.c_str()); continue; }
      std::printf("  %-8s %10.3f ms  %9.2f GFLOP/s  %6.2fx basic   (model: %.2f FLOP/B)\n", kNames[k], s * 1e3,
                  flop / s / 1e9, basic_s / s, kModelIntensity[k]);
      pmpp::bench_row(a, kNames[k]).n("radius", r).n("m", m).n("width", side).n("height", side).n("seconds", s)
          .n("gflops", flop / s / 1e9).n("speedup_vs_basic", basic_s / s).n("model_intensity", kModelIntensity[k]).emit();
    }
  }
  std::printf("\nThe book's model ceiling for each kernel is its FLOP/B times the bandwidth that lab 5.1\n"
              "measured on this device. These are measurements of this machine, not expected values.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
