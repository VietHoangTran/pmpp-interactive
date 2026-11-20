// Lab 5.2 driver: tests and benchmarks the tiled matrix multiplication kernel.
// You do not need to edit this file; your work goes in work/.
#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif

using pmpp::Buf;

// ---- entry points implemented in <variant>/ and ../shared/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeTiledMatmul(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                       id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width, uint32_t tileWidth);
void encodeNaiveMatmul(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                       id<MTLBuffer> M, id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width);

static id<MTLComputePipelineState> tiledPipeline(int tile) {
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "tiled_matmul", {{"TILE_WIDTH", std::to_string(tile)}});
}
static bool tileUsable(int tile) {
  id<MTLComputePipelineState> p = tiledPipeline(tile);
  return p && (NSUInteger)(tile * tile) <= p.maxTotalThreadsPerThreadgroup;
}
static pmpp::Timing runTiled(Buf* M, Buf* N, Buf* P, int width, int tile) {
  id<MTLComputePipelineState> p = tiledPipeline(tile);
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    encodeTiledMatmul(pass->encoder, p, (__bridge id<MTLBuffer>)pmpp::raw(M), (__bridge id<MTLBuffer>)pmpp::raw(N),
                      (__bridge id<MTLBuffer>)pmpp::raw(P), (uint32_t)width, (uint32_t)tile);
  });
}
static pmpp::Timing runNaive(Buf* M, Buf* N, Buf* P, int width) {
  id<MTLComputePipelineState> p =
      pmpp::metal::pipeline(std::string(PMPP_METAL_SHARED_DIR) + "/naive_matmul.metal", "naive_matmul");
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    encodeNaiveMatmul(pass->encoder, p, (__bridge id<MTLBuffer>)pmpp::raw(M), (__bridge id<MTLBuffer>)pmpp::raw(N),
                      (__bridge id<MTLBuffer>)pmpp::raw(P), (uint32_t)width);
  });
}
static std::vector<int> tilesToTest() { return {16, 8}; }
static std::vector<int> tilesToBench() { return {4, 8, 16, 32}; }
#else
void launchTiledMatmul(float* M, float* N, float* P, int Width);
void launchNaiveMatmul(float* M, float* N, float* P, int Width);

static bool tileUsable(int tile) { return tile * tile <= (int)pmpp::device_prop("maxThreadsPerBlock", 1024); }
static pmpp::Timing runTiled(Buf* M, Buf* N, Buf* P, int width, int /*tile: fixed at compile time*/) {
  return pmpp::run([&](void*) {
    launchTiledMatmul((float*)pmpp::raw(M), (float*)pmpp::raw(N), (float*)pmpp::raw(P), width);
  });
}
static pmpp::Timing runNaive(Buf* M, Buf* N, Buf* P, int width) {
  return pmpp::run([&](void*) {
    launchNaiveMatmul((float*)pmpp::raw(M), (float*)pmpp::raw(N), (float*)pmpp::raw(P), width);
  });
}
static std::vector<int> tilesToTest() { return {TILE_WIDTH}; }
static std::vector<int> tilesToBench() { return {TILE_WIDTH}; }
#endif

struct Problem {
  int w;
  std::vector<float> m, n, ref, mag;
  Buf *M, *N, *P;
  explicit Problem(int width, uint32_t seed) : w(width) {
    size_t e = (size_t)w * w;
    m = pmpp::random_floats(e, seed);
    n = pmpp::random_floats(e, seed + 1);
    M = pmpp::alloc(e * 4); N = pmpp::alloc(e * 4); P = pmpp::alloc(e * 4);
    pmpp::upload(M, m.data(), e * 4);
    pmpp::upload(N, n.data(), e * 4);
  }
  void reference() { pmpp::cpu_gemm(m.data(), n.data(), w, w, w, ref, &mag); }
  ~Problem() { pmpp::release(M); pmpp::release(N); pmpp::release(P); }
};

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const bool emu = std::string(pmpp::backend()) == "emu";
  for (int tile : tilesToTest()) {
    if (!tileUsable(tile)) {
      t.check("tile " + std::to_string(tile) + " usable on this device", [] { return std::string("kernel did not build, or tile*tile exceeds the device's threads-per-group limit"); });
      continue;
    }
    std::vector<int> mult = emu ? std::vector<int>{1, 2, 4} : std::vector<int>{1, 2, 4, 16, 33};
    for (int k : mult) {
      const int w = k * tile;
      t.check("tile " + std::to_string(tile) + ", width " + std::to_string(w), [&]() -> std::string {
        Problem p(w, 100 + w);
        p.reference();
        runTiled(p.M, p.N, p.P, w, tile);
        std::vector<float> got((size_t)w * w);
        pmpp::download(p.P, got.data(), got.size() * 4);
        return pmpp::describe(pmpp::compare(got.data(), p.ref.data(), p.mag.data(), got.size()), w);
      });
    }
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  if (std::string(pmpp::backend()) == "emu") {
    std::printf("bench skipped: the CPU emulator models correctness only; its timings mean nothing.\n");
    return 0;
  }
  // Grow the problem until the baseline takes long enough to time reliably.
  int w = 512;
  double naive_s = 0;
  for (;;) {
    Problem p(w, 7);
    naive_s = pmpp::time_it(a.reps, [&] { return runNaive(p.M, p.N, p.P, w); }).min_s;
    if (naive_s > 0.05 || w >= 4096) break;
    w *= 2;
  }
  const double flop = 2.0 * w * (double)w * w;
  Problem p(w, 7);
  naive_s = pmpp::time_it(a.reps, [&] { return runNaive(p.M, p.N, p.P, w); }).min_s;
  std::printf("width %d (%.2f GFLOP per multiplication), best of %d runs\n", w, flop / 1e9, a.reps);
  std::printf("  %-12s %10.4f ms  %9.2f GFLOP/s\n", "naive", naive_s * 1e3, flop / naive_s / 1e9);
  pmpp::bench_row(a, "naive").n("width", w).n("seconds", naive_s).n("gflops", flop / naive_s / 1e9).emit();
  for (int tile : tilesToBench()) {
    if (w % tile != 0 || !tileUsable(tile)) continue;
    double s = pmpp::time_it(a.reps, [&] { return runTiled(p.M, p.N, p.P, w, tile); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  tiled, T=%-3d  failed: %s\n", tile, err.c_str()); continue; }
    std::printf("  tiled, T=%-3d %10.4f ms  %9.2f GFLOP/s   %5.2fx naive\n", tile, s * 1e3, flop / s / 1e9, naive_s / s);
    pmpp::bench_row(a, "tiled").n("tile", tile).n("width", w).n("seconds", s)
        .n("gflops", flop / s / 1e9).n("speedup_vs_naive", naive_s / s).emit();
  }
  return 0;
}

int main(int argc, char** argv) {
  pmpp::Args a = pmpp::parse_args(argc, argv);
  if (a.device) { pmpp::emit_device(); return 0; }
  std::string why;
  if (!pmpp::device_available(&why)) { std::printf("no %s device: %s\n", pmpp::backend(), why.c_str()); return 3; }
  int rc = 0;
  if (a.test) rc |= tests(a);
  if (a.bench) rc |= bench(a);
  return rc;
}
