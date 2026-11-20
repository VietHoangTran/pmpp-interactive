// Lab 6.3 driver: tests and benchmarks your tiled transpose at two shared-memory
// pitches, and runs the two provided instruments: a bank probe (one warp, lanes
// `stride` words apart) and a plain copy as the yardstick for the transpose.
// You do not need to edit this file; your work goes in work/.
#include <limits>

#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
// The TILE in CMakeLists.txt. On CUDA and HIP the build with this tile also runs the
// instruments; the extra build (TILE_WIDTH 32) only times its transpose, so that the
// instrument rows are not recorded twice.
#define LAB_DEFAULT_TILE 16

using pmpp::Buf;

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeTiledTranspose(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                          id<MTLBuffer> out, int32_t rows, int32_t cols, int32_t pitch, uint32_t tileWidth);
void encodeBankProbe(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> out,
                     uint32_t lanes, int32_t stride, int32_t iters, float zero);
void encodeCopy(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> out, int32_t n);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.rfind('/')); }
static std::string provided(const char* file) { return dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/" + file; }
static id<MTLComputePipelineState> transposePipeline(int tile) {
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "tiled_transpose", {{"TILE_WIDTH", std::to_string(tile)}});
}
static bool kernelBuilds(int tile) { return transposePipeline(tile) != nil; }
static bool tileFits(int tile) {
  id<MTLComputePipelineState> p = transposePipeline(tile);
  return p && (NSUInteger)(tile * tile) <= p.maxTotalThreadsPerThreadgroup;
}
static pmpp::Timing runTranspose(Buf* in, Buf* out, int rows, int cols, int pitch, int tile) {
  id<MTLComputePipelineState> p = transposePipeline(tile);
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeTiledTranspose(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), rows, cols, pitch, (uint32_t)tile);
  });
}
static id<MTLComputePipelineState> probePipeline() { return pmpp::metal::pipeline(provided("bank_probe.metal"), "bank_probe"); }
// The SIMD-group width of the probe's own pipeline (notes/today/ch05-facts.md C3).
static int probeLanes() { id<MTLComputePipelineState> p = probePipeline(); return p ? (int)p.threadExecutionWidth : 0; }
static const char* kWidthSource = "threadExecutionWidth of the probe's pipeline";
static pmpp::Timing runProbe(Buf* out, int lanes, int stride, int iters) {
  id<MTLComputePipelineState> p = probePipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeBankProbe(((pmpp::metal::Pass*)v)->encoder, p, buffer(out), (uint32_t)lanes, stride, iters, 0.0f);
  });
}
static pmpp::Timing runCopy(Buf* in, Buf* out, int n) {
  id<MTLComputePipelineState> p = pmpp::metal::pipeline(provided("copy.metal"), "copy_floats");
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeCopy(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), n); });
}
static std::vector<int> tiles() { return {16, 32}; }   // Metal compiles the tile width at run time
static bool runsInstruments() { return true; }
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchTiledTranspose(const float* in, float* out, int rows, int cols, int pitch);
void launchBankProbe(float* out, int lanes, int stride, int iters, float zero);
void launchCopy(const float* in, float* out, int n);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static bool kernelBuilds(int) { return true; }   // it was compiled into this executable
static bool tileFits(int tile) { return tile * tile <= (int)pmpp::device_prop("maxThreadsPerBlock", 1024); }
static pmpp::Timing runTranspose(Buf* in, Buf* out, int rows, int cols, int pitch, int /*tile: fixed at compile time*/) {
  return pmpp::run([&](void*) { launchTiledTranspose(f(in), f(out), rows, cols, pitch); });
}
static int probeLanes() { return (int)pmpp::device_prop("warpSize", 0); }
static const char* kWidthSource = "warpSize from the device properties";
static pmpp::Timing runProbe(Buf* out, int lanes, int stride, int iters) {
  return pmpp::run([&](void*) { launchBankProbe(f(out), lanes, stride, iters, 0.0f); });
}
static pmpp::Timing runCopy(Buf* in, Buf* out, int n) { return pmpp::run([&](void*) { launchCopy(f(in), f(out), n); }); }
static std::vector<int> tiles() { return {TILE_WIDTH}; }
static bool runsInstruments() { return TILE_WIDTH == LAB_DEFAULT_TILE; }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

static const float kNaN = std::numeric_limits<float>::quiet_NaN();

// in: rows x cols of random values; out: cols x rows, poisoned with NaN so that an
// element your kernel never writes is reported.
struct TransposeProblem {
  int rows, cols;
  std::vector<float> in;
  Buf *In, *Out;
  TransposeProblem(int r, int c, uint32_t seed, bool withData = true) : rows(r), cols(c) {
    const size_t n = (size_t)r * c;
    In = pmpp::alloc(n * 4); Out = pmpp::alloc(n * 4);
    if (withData) {
      in = pmpp::random_floats(n, seed);
      pmpp::upload(In, in.data(), n * 4);
      std::vector<float> poison(n, kNaN);
      pmpp::upload(Out, poison.data(), n * 4);
    }
  }
  TransposeProblem(const TransposeProblem&) = delete;
  ~TransposeProblem() { pmpp::release(In); pmpp::release(Out); }

  // A transpose only moves values, so the comparison is exact.
  std::string check() const {
    const size_t n = (size_t)rows * cols;
    std::vector<float> got(n), ref(n);
    pmpp::download(Out, got.data(), n * 4);
    for (int r = 0; r < rows; ++r)
      for (int c = 0; c < cols; ++c) ref[(size_t)c * rows + r] = in[(size_t)r * cols + c];
    pmpp::Cmp cmp = pmpp::compare(got.data(), ref.data(), nullptr, n, 0.0);
    if (cmp.ok) return "";
    const size_t i = cmp.first_bad, orow = i / rows, ocol = i % rows;
    char b[240];
    std::snprintf(b, sizeof b, "%zu of %zu elements wrong; first at out[%zu][%zu] (should be in[%zu][%zu]): got %g, expected %g",
                  cmp.bad, n, orow, ocol, ocol, orow, (double)got[i], (double)ref[i]);
    return b;
  }
};

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);

  if (runsInstruments()) {
    t.check("provided copy, n = 1000", [&]() -> std::string {
      const int n = 1000;
      auto x = pmpp::random_floats(n, 3);
      std::vector<float> poison(n, kNaN), got(n);
      Buf *I = pmpp::alloc(n * 4), *O = pmpp::alloc(n * 4);
      pmpp::upload(I, x.data(), n * 4); pmpp::upload(O, poison.data(), n * 4);
      runCopy(I, O, n);
      pmpp::download(O, got.data(), n * 4);
      pmpp::release(I); pmpp::release(O);
      return pmpp::describe(pmpp::compare(got.data(), x.data(), nullptr, n, 0.0), n);
    });
    const int lanes = probeLanes();
    t.check("provided bank probe, stride 1, one warp of " + std::to_string(lanes) + " lanes", [&]() -> std::string {
      if (lanes <= 0 || lanes > 1024) return "the device reports no usable warp width (" + std::string(kWidthSource) + ")";
      const int iters = 100, stride = 1;
      std::vector<float> got(lanes, kNaN), ref(lanes);
      Buf* O = pmpp::alloc((size_t)lanes * 4);
      pmpp::upload(O, got.data(), (size_t)lanes * 4);
      runProbe(O, lanes, stride, iters);
      pmpp::download(O, got.data(), (size_t)lanes * 4);
      pmpp::release(O);
      for (int l = 0; l < lanes; ++l) ref[l] = (float)((iters + 1) * ((l * stride) & 1023));
      return pmpp::describe(pmpp::compare(got.data(), ref.data(), nullptr, lanes, 0.0), lanes);
    });
  }

  struct Shape { int rows, cols; };
  std::vector<Shape> shapes = {{1, 1}, {7, 5}, {16, 16}, {17, 33}, {64, 48}, {100, 1}, {1, 100}};
  if (!pmpp::is_emu()) { shapes.push_back({1000, 1000}); shapes.push_back({513, 2049}); }

  for (int tile : tiles()) {
    if (!kernelBuilds(tile)) {
      t.check("tile " + std::to_string(tile) + ": your kernel builds", [] { return std::string("tiled_transpose did not compile; the message is above"); });
      continue;
    }
    if (!tileFits(tile)) {
      std::printf("  (tile %d skipped: %d threads per threadgroup is more than this pipeline accepts)\n", tile, tile * tile);
      continue;
    }
    for (int pitch : {tile, tile + 1}) {
      for (const Shape& s : shapes) {
        char name[120];
        std::snprintf(name, sizeof name, "tile %d, pitch %d (%s), %d x %d", tile, pitch,
                      pitch == tile ? "unpadded" : "padded", s.rows, s.cols);
        t.check(name, [&]() -> std::string {
          TransposeProblem p(s.rows, s.cols, 200 + (uint32_t)(s.rows * 7 + s.cols));
          runTranspose(p.In, p.Out, s.rows, s.cols, pitch, tile);
          return p.check();
        });
      }
    }
  }
  return t.finish();
}

// Shared-memory strides the probe visits: Exercise 5's seven (32, 31, 24, 16, 12, 8, 7)
// plus 1, 2, 4 and 33.
static const int kStrides[] = {1, 2, 4, 7, 8, 12, 16, 24, 31, 32, 33};

static void benchInstruments(const pmpp::Args& a) {
  const int lanes = probeLanes();
  std::printf("SIMD width: %d (%s)\n", lanes, kWidthSource);
  pmpp::bench_row(a, "simd_width").n("width", lanes).emit();
  if (lanes <= 0 || lanes > 1024) { std::printf("bank probe skipped: no usable width\n"); return; }

  Buf* O = pmpp::alloc((size_t)lanes * 4);
  // Grow the step count until stride 1 takes long enough to time reliably.
  int iters = 1 << 12;
  double s1 = 0;
  for (;;) {
    s1 = pmpp::time_it(a.reps, [&] { return runProbe(O, lanes, 1, iters); }).min_s;
    if (s1 > 0.01 || iters >= (1 << 24)) break;
    iters *= 2;
  }
  std::printf("Bank probe: one warp of %d lanes, %d steps of one read and one write each, best of %d runs\n",
              lanes, iters, a.reps);
  std::printf("  %6s %10s %15s %12s\n", "stride", "time ms", "ns per access", "vs stride 1");
  for (int stride : kStrides) {
    const double s = stride == 1 ? s1 : pmpp::time_it(a.reps, [&] { return runProbe(O, lanes, stride, iters); }).min_s;
    const std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  %6d failed: %s\n", stride, err.c_str()); continue; }
    const double ns = s / (2.0 * iters) * 1e9;   // an access is one warp-wide read or write
    std::printf("  %6d %10.3f %15.3f %11.2fx\n", stride, s * 1e3, ns, s / s1);
    pmpp::bench_row(a, "bank_probe").n("stride", stride).n("lanes", lanes).n("iters", iters).n("seconds", s)
        .n("ns_per_access", ns).n("relative_to_stride1", s / s1).emit();
  }
  pmpp::release(O);
}

static int bench(const pmpp::Args& a) {
  if (runsInstruments()) benchInstruments(a);
  else std::printf("(the bank probe and the copy are measured by the TILE_WIDTH=%d build)\n", LAB_DEFAULT_TILE);

  // Square matrices. Grow the side from 1024 until the copy takes over 5 ms, up to 8192,
  // keeping the two arrays within a quarter of the memory the device reports.
  const double mem = deviceMemory();
  auto fits = [&](long long side) { return mem <= 0 || 8.0 * (double)side * side <= mem / 4; };
  int side = 1024;
  double copy_s = 0;
  for (;;) {
    TransposeProblem p(side, side, 5, false);
    copy_s = pmpp::time_it(a.reps, [&] { return runCopy(p.In, p.Out, side * side); }).min_s;
    if (copy_s > 0.005 || side >= 8192 || !fits(2LL * side)) break;
    side *= 2;
  }
  TransposeProblem p(side, side, 5);
  const int n = side * side;
  const double bytes = 8.0 * n;   // every element read once and written once
  copy_s = pmpp::time_it(a.reps, [&] { return runCopy(p.In, p.Out, n); }).min_s;
  std::printf("Copy of %d x %d floats (%.0f MB moved): %.4f ms, %.1f GB/s, the yardstick for the transpose\n",
              side, side, bytes / 1e6, copy_s * 1e3, bytes / copy_s / 1e9);
  if (runsInstruments())
    pmpp::bench_row(a, "copy").n("n", n).n("seconds", copy_s).n("gb_per_s", bytes / copy_s / 1e9).emit();

  std::printf("Transpose of %d x %d, best of %d runs\n", side, side, a.reps);
  std::printf("  %4s %6s %10s %9s %9s\n", "tile", "pitch", "time ms", "GB/s", "vs copy");
  for (int tile : tiles()) {
    if (!tileFits(tile)) { std::printf("  %4d  skipped: the kernel did not build, or %d threads per group is too many here\n", tile, tile * tile); continue; }
    for (int pitch : {tile, tile + 1}) {
      const double s = pmpp::time_it(a.reps, [&] { return runTranspose(p.In, p.Out, side, side, pitch, tile); }).min_s;
      const std::string err = pmpp::take_error();
      if (!err.empty() || s <= 0) { std::printf("  %4d %6d failed: %s\n", tile, pitch, err.c_str()); continue; }
      std::printf("  %4d %6d %10.4f %9.1f %8.2fx\n", tile, pitch, s * 1e3, bytes / s / 1e9, copy_s / s);
      pmpp::bench_row(a, "transpose").n("tile", tile).n("pitch", pitch).n("rows", side).n("cols", side)
          .n("seconds", s).n("gb_per_s", bytes / s / 1e9).emit();
    }
  }
  std::printf("These are measurements of this machine. 'vs copy' is the copy's time over the transpose's.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
