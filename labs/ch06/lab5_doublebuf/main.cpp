// Lab 6.5 driver: tests the double-buffered tiled matrix multiplication and times it
// against lab 5.2's two-barrier kernel (provided/). You do not need to edit this
// file; your work goes in work/.
//
// How a missing barrier shows up on each backend:
//   emu     One emulated thread runs at a time, from one barrier to the next, under
//           three fixed orders. A thread that overwrites a tile slot before its
//           neighbours have read it does so in every run: wrong values, every time.
//   race    The threads run concurrently under ThreadSanitizer. A write and a read
//           of the same shared slot with no barrier between them are reported with
//           both source lines, whether or not that run's values came out wrong.
//           ThreadSanitizer reports each pair of lines once per run, so a later
//           case of the same kernel can pass.
//   device  Each case is launched several times on the same inputs, with P refilled
//           with NaN before every launch. A failing case says in how many launches
//           it was wrong. A case that passed every launch is evidence, not proof.
#include "pmpp/harness.hpp"

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
using pmpp::Buf;

enum class Kernel { TwoBarriers, DoubleBuffered };
static const char* kernelName(Kernel k) { return k == Kernel::TwoBarriers ? "two_barriers" : "double_buffered"; }
static const char* kernelLabel(Kernel k) {
  return k == Kernel::TwoBarriers ? "two barriers (provided)" : "double buffered";
}

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeDoubleBufferedMatmul(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M,
                                id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width, uint32_t tileWidth);
void encodeTwoBarrierMatmul(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M,
                            id<MTLBuffer> N, id<MTLBuffer> P, uint32_t width, uint32_t tileWidth);

static std::string dirOf(const std::string& p) {
  const size_t s = p.find_last_of('/');
  return s == std::string::npos ? std::string(".") : p.substr(0, s);
}
// Metal compiles kernels at run time, so one program can try several tile widths.
static id<MTLComputePipelineState> pipelineFor(Kernel k, int tile) {
  const std::vector<std::pair<std::string, std::string>> macros = {{"TILE_WIDTH", std::to_string(tile)}};
  if (k == Kernel::DoubleBuffered) return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "double_buffered_matmul", macros);
  return pmpp::metal::pipeline(dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/tiled_two_barriers.metal",
                               "two_barrier_matmul", macros);
}
// Empty when kernel `k` can run with this tile width on this device; otherwise why not.
// The limits are the ones this pipeline and this device report.
static std::string tileProblem(Kernel k, int tile) {
  char b[320];
  id<MTLComputePipelineState> p = pipelineFor(k, tile);
  if (!p) {
    std::snprintf(b, sizeof b, "%s did not compile with TILE_WIDTH=%d: %s", kernelLabel(k), tile,
                  pmpp::take_error().c_str());
    return b;
  }
  if ((NSUInteger)(tile * tile) > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "%d x %d threads exceed this pipeline's maxTotalThreadsPerThreadgroup (%lu)", tile,
                  tile, (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  const NSUInteger have = pmpp::metal::device().maxThreadgroupMemoryLength;
  if (p.staticThreadgroupMemoryLength > have) {
    std::snprintf(b, sizeof b, "the tiles take %lu bytes of threadgroup memory; this device's maxThreadgroupMemoryLength is %lu",
                  (unsigned long)p.staticThreadgroupMemoryLength, (unsigned long)have);
    return b;
  }
  return "";
}
static pmpp::Timing runKernel(Kernel k, Buf* M, Buf* N, Buf* P, int width, int tile) {
  id<MTLComputePipelineState> p = pipelineFor(k, tile);
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    auto encode = k == Kernel::DoubleBuffered ? encodeDoubleBufferedMatmul : encodeTwoBarrierMatmul;
    encode(pass->encoder, p, pmpp::metal::buffer(M), pmpp::metal::buffer(N), pmpp::metal::buffer(P),
           (uint32_t)width, (uint32_t)tile);
  });
}
// 16 must work. 32 is tried too: 1024 threads per threadgroup, and two pairs of
// 32 x 32 float tiles. Where the device does not take it, it is skipped with a note.
static const std::vector<int> kTiles = {16, 32};
static const int kRequiredTile = 16;
#else
void launchDoubleBufferedMatmul(float* M, float* N, float* P, int Width);
void launchTwoBarrierMatmul(float* M, float* N, float* P, int Width);

static std::string tileProblem(Kernel k, int tile) {
  char b[240];
  const double maxThreads = pmpp::device_prop("maxThreadsPerBlock", 0);
  const double maxShared = pmpp::device_prop("sharedMemPerBlock", 0);
  const double need = (k == Kernel::DoubleBuffered ? 4.0 : 2.0) * tile * tile * sizeof(float);
  if (maxThreads > 0 && tile * tile > maxThreads) {
    std::snprintf(b, sizeof b, "%d x %d threads exceed this device's maxThreadsPerBlock (%.0f)", tile, tile, maxThreads);
    return b;
  }
  if (maxShared > 0 && need > maxShared) {
    std::snprintf(b, sizeof b, "the tiles take %.0f bytes of shared memory; this device's sharedMemPerBlock is %.0f",
                  need, maxShared);
    return b;
  }
  return "";
}
static pmpp::Timing runKernel(Kernel k, Buf* M, Buf* N, Buf* P, int width, int /*tile: fixed when compiled*/) {
  float* m = (float*)pmpp::raw(M);
  float* n = (float*)pmpp::raw(N);
  float* p = (float*)pmpp::raw(P);
  return pmpp::run([&](void*) {
    if (k == Kernel::DoubleBuffered) launchDoubleBufferedMatmul(m, n, p, width);
    else launchTwoBarrierMatmul(m, n, p, width);
  });
}
// The tile width is fixed when the CUDA-family program is compiled. The build also
// makes 8 and 32 variants, which the benchmark runs as separate programs.
static const std::vector<int> kTiles = {TILE_WIDTH};
static const int kRequiredTile = TILE_WIDTH;
#endif

// ---- tests ----
// Widths are multiples of the tile width; the number of phases is width / tile.
static std::vector<int> testWidths(int tile) {
#if defined(PMPP_BACKEND_EMU)
  const std::vector<int> want = {16, 32, 64};
#elif defined(PMPP_BACKEND_RACE)
  const std::vector<int> want = {32, 48};   // ThreadSanitizer is slow: keep it small
#else
  const std::vector<int> want = {16, 32, 256, 33 * tile};
#endif
  std::vector<int> out;
  for (int w : want)
    if (w % tile == 0) out.push_back(w);
  return out;
}

// Real devices give a different schedule on every launch, so each case is launched
// several times. The emulators run each case once per schedule (Tests does that).
static int launchesPerCase() { return pmpp::is_emu() ? 1 : 10; }

// Runs kernel `k` `launches` times on one problem. Empty string = correct every time.
static std::string checkKernel(Kernel k, int w, int tile, int launches) {
  pmpp::GemmProblem p(w, w, w, 600 + w);
  const std::vector<float> nan((size_t)w * w, std::nanf(""));
  int wrong = 0;
  std::string first;
  for (int i = 0; i < launches; ++i) {
    pmpp::upload(p.C, nan.data(), nan.size() * 4);   // an element the kernel never writes stays NaN
    runKernel(k, p.A, p.B, p.C, w, tile);
    std::string r = p.check();
    if (!r.empty() && wrong++ == 0) first = r;
  }
  std::string out;
  if (wrong) {
    out = launches == 1 ? first
                        : "wrong in " + std::to_string(wrong) + " of " + std::to_string(launches) +
                              " launches; first: " + first;
  }
  // A launch error, a compile error, or (backend race) a ThreadSanitizer report.
  const std::string err = pmpp::take_error();
  if (!err.empty()) out += (out.empty() ? "" : "; ") + err;
  return out;
}

static std::string phases(int w, int tile) {
  const int n = w / tile;
  return std::to_string(n) + (n == 1 ? " phase" : " phases");
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const int launches = launchesPerCase();
  for (int tile : kTiles) {
    const std::string T = "tile " + std::to_string(tile);
    std::string why = tileProblem(Kernel::DoubleBuffered, tile);
    if (why.empty()) why = tileProblem(Kernel::TwoBarriers, tile);
    if (!why.empty()) {
      if (tile == kRequiredTile) t.check(T + " usable on this device", [&] { return why; });
      else std::printf("  note: %s skipped on this device: %s\n", T.c_str(), why.c_str());
      continue;
    }
    // The provided baseline once, so that the benchmark compares two correct kernels.
    const int wb = 2 * tile;
    t.check("provided two-barrier kernel, " + T + ", width " + std::to_string(wb),
            [&] { return checkKernel(Kernel::TwoBarriers, wb, tile, 1); });
    for (int w : testWidths(tile)) {
      t.check("double buffered, " + T + ", width " + std::to_string(w) + " (" + phases(w, tile) + ")",
              [&] { return checkKernel(Kernel::DoubleBuffered, w, tile, launches); });
    }
  }
  return t.finish();
}

// ---- benchmark (real devices only; lab_main skips it on the emulators) ----
// The spec's rows name the kernel in "variant"; "kernel" repeats it because the
// verify report hides a column called "variant". The result these rows belong to
// still says whether they measured your work or the reference solution.
static void emitRow(const pmpp::Args& a, Kernel k, int tile, int w, double s, double flop, double baseline_s) {
  pmpp::Json j;
  j.s("kind", "bench").s("lab", a.lab).s("variant", kernelName(k)).s("backend", pmpp::backend()).s("name", "matmul");
  j.s("kernel", kernelName(k)).n("tile", tile).n("width", w).n("seconds", s).n("gflops", flop / s / 1e9)
      .n("speedup_vs_two_barriers", baseline_s / s);
  j.emit();
}

static int bench(const pmpp::Args& a) {
  std::string why = tileProblem(Kernel::TwoBarriers, kRequiredTile);
  if (!why.empty()) { std::printf("bench: the baseline cannot run: %s\n", why.c_str()); return 1; }
  // Grow the problem until the baseline takes long enough to time reliably (as lab 5.2).
  int w = 512;
  for (;;) {
    pmpp::GemmProblem p(w, w, w, 7);
    const double s = pmpp::time_it(a.reps, [&] { return runKernel(Kernel::TwoBarriers, p.A, p.B, p.C, w, kRequiredTile); }).min_s;
    if (s > 0.05 || w >= 4096) break;
    w *= 2;
  }
  pmpp::GemmProblem p(w, w, w, 7);
  const double flop = p.flop();
  // The two kernels alternate, twice each, so that a clock change during the run
  // does not favour whichever went first. Each figure is the best of 2 x reps runs.
  std::printf("width %d (%.2f GFLOP per multiplication), best of %d runs per kernel\n", w, flop / 1e9, 2 * a.reps);
  int rc = 0;
  for (int tile : kTiles) {
    if (w % tile != 0) continue;
    why = tileProblem(Kernel::TwoBarriers, tile);
    if (why.empty()) why = tileProblem(Kernel::DoubleBuffered, tile);
    if (!why.empty()) { std::printf("  T=%-3d skipped: %s\n", tile, why.c_str()); continue; }
    double s2 = 1e300, s1 = 1e300;
    for (int round = 0; round < 2; ++round) {
      s2 = std::min(s2, pmpp::time_it(a.reps, [&] { return runKernel(Kernel::TwoBarriers, p.A, p.B, p.C, w, tile); }).min_s);
      s1 = std::min(s1, pmpp::time_it(a.reps, [&] { return runKernel(Kernel::DoubleBuffered, p.A, p.B, p.C, w, tile); }).min_s);
    }
    const std::string err = pmpp::take_error();
    if (!err.empty() || s1 <= 0 || s2 <= 0) {
      std::printf("  T=%-3d failed: %s\n", tile, err.empty() ? "no time was recorded" : err.c_str());
      rc = 1;
      continue;
    }
    std::printf("  T=%-3d %-24s %10.4f ms  %9.2f GFLOP/s\n", tile, kernelLabel(Kernel::TwoBarriers), s2 * 1e3,
                flop / s2 / 1e9);
    std::printf("  T=%-3d %-24s %10.4f ms  %9.2f GFLOP/s   %5.3fx the two-barrier kernel\n", tile,
                kernelLabel(Kernel::DoubleBuffered), s1 * 1e3, flop / s1 / 1e9, s2 / s1);
    emitRow(a, Kernel::TwoBarriers, tile, w, s2, flop, s2);
    emitRow(a, Kernel::DoubleBuffered, tile, w, s1, flop, s2);
  }
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
