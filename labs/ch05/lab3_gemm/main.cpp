// Lab 5.3 driver: tests and benchmarks the general (rectangular, any size)
// tiled matrix multiplication. Your work goes in work/.
#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
using pmpp::Buf;

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeGemm(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A, id<MTLBuffer> B,
                id<MTLBuffer> C, uint32_t j, uint32_t k, uint32_t l, uint32_t tileWidth);
static pmpp::Timing runGemm(pmpp::GemmProblem& p) {
  auto pso = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gemm", {{"TILE_WIDTH", std::to_string(TILE_WIDTH)}});
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeGemm(((pmpp::metal::Pass*)v)->encoder, pso, pmpp::metal::buffer(p.A), pmpp::metal::buffer(p.B),
               pmpp::metal::buffer(p.C), (uint32_t)p.j, (uint32_t)p.k, (uint32_t)p.l, TILE_WIDTH);
  });
}
#else
void launchGemm(float* A, float* B, float* C, int j, int k, int l);
static pmpp::Timing runGemm(pmpp::GemmProblem& p) {
  return pmpp::run([&](void*) {
    launchGemm((float*)pmpp::raw(p.A), (float*)pmpp::raw(p.B), (float*)pmpp::raw(p.C), p.j, p.k, p.l);
  });
}
#endif

struct Shape { int j, k, l; const char* why; };

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::vector<Shape> shapes = {
      {3, 3, 3, "the book's 3x3 example with a wider tile"},
      {1, 1, 1, "single element"},
      {16, 16, 16, "exactly one tile"},
      {32, 48, 16, "rectangular, all multiples of the tile"},
      {17, 33, 9, "nothing is a multiple of the tile"},
      {1, 40, 7, "one row: a single token through a linear layer"},
      {5, 1, 9, "inner dimension 1 (outer product)"},
      {33, 65, 50, "several partial tiles in every direction"},
  };
  if (!pmpp::is_emu()) {
    shapes.push_back({4, 300, 1000, "a small batch into a wide layer"});
    shapes.push_back({257, 129, 511, "large and awkward"});
  }
  for (const Shape& s : shapes) {
    char name[160];
    std::snprintf(name, sizeof name, "%d x %d times %d x %d  (%s)", s.j, s.k, s.k, s.l, s.why);
    t.check(name, [&]() -> std::string {
      pmpp::GemmProblem p(s.j, s.k, s.l, 1000 + s.j * 31 + s.k * 7 + s.l);
      // Poison C so that elements the kernel never writes are detected.
      std::vector<float> poison((size_t)s.j * s.l, 12345.0f);
      pmpp::upload(p.C, poison.data(), poison.size() * 4);
      runGemm(p);
      return p.check();
    });
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // The book's own example layer (section 20.6): 1024 inputs -> 4096 outputs.
  // The batch size j is the number of rows pushed through it at once.
  const int k = 1024, l = 4096;
  std::printf("linear layer %d -> %d, tile %d, best of %d runs\n", k, l, TILE_WIDTH, a.reps);
  std::printf("  %6s %12s %14s %16s\n", "batch", "time (ms)", "GFLOP/s", "rows per second");
  for (int j : {1, 4, 16, 64, 256, 1024}) {
    pmpp::GemmProblem p(j, k, l, 3, false);
    double s = pmpp::time_it(a.reps, [&] { return runGemm(p); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  failed: %s\n", err.c_str()); return 1; }
    std::printf("  %6d %12.4f %14.2f %16.0f\n", j, s * 1e3, p.flop() / s / 1e9, j / s);
    pmpp::bench_row(a, "linear_layer").n("batch", j).n("k", k).n("l", l).n("tile", TILE_WIDTH)
        .n("seconds", s).n("gflops", p.flop() / s / 1e9).n("rows_per_s", j / s).emit();
  }
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
