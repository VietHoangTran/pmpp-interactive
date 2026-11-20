// Lab 5.4 driver: tests a tiled, boundary-checked matrix multiplication kernel
// that ships with three defects. Your work goes in work/.
#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
using pmpp::Buf;

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeTiledMatmul(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M, id<MTLBuffer> N,
                       id<MTLBuffer> P, uint32_t width, uint32_t tileWidth);
static pmpp::Timing runIt(pmpp::GemmProblem& p) {
  auto pso = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "tiled_matmul", {{"TILE_WIDTH", std::to_string(TILE_WIDTH)}});
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeTiledMatmul(((pmpp::metal::Pass*)v)->encoder, pso, pmpp::metal::buffer(p.A), pmpp::metal::buffer(p.B),
                      pmpp::metal::buffer(p.C), (uint32_t)p.j, TILE_WIDTH);
  });
}
#else
void launchTiledMatmul(float* M, float* N, float* P, int Width);
static pmpp::Timing runIt(pmpp::GemmProblem& p) {
  return pmpp::run([&](void*) {
    launchTiledMatmul((float*)pmpp::raw(p.A), (float*)pmpp::raw(p.B), (float*)pmpp::raw(p.C), p.j);
  });
}
#endif

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::vector<int> widths = pmpp::is_emu() ? std::vector<int>{16, 32, 3, 24, 40, 50}
                                           : std::vector<int>{16, 32, 3, 24, 40, 50, 200, 333};
  for (int w : widths) {
    t.check("width " + std::to_string(w) + (w % TILE_WIDTH ? "  (not a multiple of the tile)" : "  (multiple of the tile)"),
            [&]() -> std::string {
              // Run twice on fresh data: stale shared-memory contents from the first
              // run are what an un-zeroed slot would pick up in the second.
              std::string r;
              for (int rep = 0; rep < 2 && r.empty(); ++rep) {
                pmpp::GemmProblem p(w, w, w, 500 + w + rep * 17);
                runIt(p);
                r = p.check();
              }
              return r;
            });
  }
  return t.finish();
}

static int bench(const pmpp::Args&) { std::printf("this lab has no benchmark.\n"); return 0; }

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
