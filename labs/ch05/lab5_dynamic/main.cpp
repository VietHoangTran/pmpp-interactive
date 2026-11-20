// Lab 5.5 driver: tiled matrix multiplication with a tile width chosen at run
// time from what the device reports. Your work goes in work/.
#include "pmpp/harness.hpp"
using pmpp::Buf;

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeTiledMatmulDynamic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M,
                              id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width, uint32_t tileWidth);
static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "tiled_matmul_dynamic"); }
static pmpp::Timing runIt(pmpp::GemmProblem& p, int tile) {
  auto s = pso();
  if (!s) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeTiledMatmulDynamic(((pmpp::metal::Pass*)v)->encoder, s, pmpp::metal::buffer(p.A), pmpp::metal::buffer(p.B),
                             pmpp::metal::buffer(p.C), (uint32_t)p.j, (uint32_t)tile);
  });
}
// The limits this device reports: threads per group and bytes of threadgroup memory.
static double maxThreads() { auto s = pso(); return s ? (double)s.maxTotalThreadsPerThreadgroup : 0; }
static double maxSharedBytes() { return pmpp::device_prop("maxThreadgroupMemoryLength", 0); }
#else
void launchTiledMatmulDynamic(float* M, float* N, float* P, int Width, int tileWidth);
static pmpp::Timing runIt(pmpp::GemmProblem& p, int tile) {
  return pmpp::run([&](void*) {
    launchTiledMatmulDynamic((float*)pmpp::raw(p.A), (float*)pmpp::raw(p.B), (float*)pmpp::raw(p.C), p.j, tile);
  });
}
static double maxThreads() { return pmpp::device_prop("maxThreadsPerBlock", 0); }
static double maxSharedBytes() { return pmpp::device_prop("sharedMemPerBlock", 0); }
#endif

// Largest power-of-two tile width whose block fits both device limits.
static int largestTile() {
  int t = 1;
  while ((double)(2 * t) * (2 * t) <= maxThreads() && 2.0 * (2 * t) * (2 * t) * sizeof(float) <= maxSharedBytes()) t *= 2;
  return t;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int big = largestTile();
  std::printf("device limits: %.0f threads per block, %.0f bytes of shared memory per block -> largest tile %d\n",
              maxThreads(), maxSharedBytes(), big);
  std::vector<int> tiles = {4, 8, 16};
  if (big > 16) tiles.push_back(big);
  for (int tile : tiles) {
    if (tile > big) continue;
    std::vector<int> widths = pmpp::is_emu() ? std::vector<int>{tile, 2 * tile, 3, 2 * tile + 5}
                                             : std::vector<int>{tile, 2 * tile, 3, 2 * tile + 5, 200, 333};
    for (int w : widths) {
      t.check("tile " + std::to_string(tile) + ", width " + std::to_string(w), [&]() -> std::string {
        pmpp::GemmProblem p(w, w, w, 900 + tile * 13 + w);
        std::vector<float> poison((size_t)w * w, 12345.0f);
        pmpp::upload(p.C, poison.data(), poison.size() * 4);
        runIt(p, tile);
        return p.check();
      });
    }
  }
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  const int w = 2048;
  pmpp::GemmProblem p(w, w, w, 5, false);
  std::printf("width %d, one binary, tile width chosen at run time, best of %d runs\n", w, a.reps);
  for (int tile = 4; tile <= largestTile(); tile *= 2) {
    double s = pmpp::time_it(a.reps, [&] { return runIt(p, tile); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  T=%-3d failed: %s\n", tile, err.c_str()); continue; }
    std::printf("  T=%-3d %10.4f ms  %9.2f GFLOP/s  (%d bytes of shared memory per block)\n", tile, s * 1e3,
                p.flop() / s / 1e9, 2 * tile * tile * 4);
    pmpp::bench_row(a, "tiled_dynamic").n("tile", tile).n("width", w).n("seconds", s)
        .n("gflops", p.flop() / s / 1e9).n("shared_bytes_per_block", 2.0 * tile * tile * 4).emit();
  }
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
