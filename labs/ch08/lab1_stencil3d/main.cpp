// Lab 8.1 driver: tests and benchmarks the 3D 7-point stencil kernels.
// You do not need to edit this file; your work goes in work/.
//
// Kernels exercised
//   yours     basic    one thread per point (book Fig. 8.6)
//             regtile  thread coarsening in z with register tiling (Fig. 8.12)
//   provided  cube     shared-memory cube tile, 8 x 8 x 8 (Fig. 8.8), ../shared/tiled_cube.*
//             planes   three shared planes, slab depth T - 2 (Fig. 8.10), ../shared/planes.*
//
// Before every launch the driver copies `in` into `out` and puts a guard region
// after `out`. A correct kernel changes only the interior points of `out`.
#include <cstring>
#include <map>
#include <memory>

#include "pmpp/harness.hpp"
#include "stencil.h"

#ifndef TILE_WIDTH
#define TILE_WIDTH 32
#endif

using pmpp::Buf;

// The weights every test uses. All seven are different, so that a kernel pairing a
// weight with the wrong neighbour (c3 with y+1, say) gives a wrong answer.
static const StencilCoeffs kCoeffs = {0.40f, 0.11f, 0.13f, 0.07f, 0.17f, 0.05f, 0.19f};

enum class Kind { Basic, RegTile, Cube, Planes };
struct Kernel {
  Kind kind;
  int tile = 0;           // threads per side of the block (RegTile, Planes); Cube is always 8
  unsigned depth = 0;     // output planes per slab (RegTile)
};

static int threadsNeeded(const Kernel& k) {
  switch (k.kind) {
    case Kind::Basic: return 8 * 8 * 8;
    case Kind::Cube: return 8 * 8 * 8;
    default: return k.tile * k.tile;
  }
}

// ---- entry points implemented in <variant>/ and ../shared/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeStencilBasic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                        id<MTLBuffer> out, uint32_t N, StencilCoeffs c);
void encodeStencilRegTiled(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                           id<MTLBuffer> out, uint32_t N, uint32_t depth, uint32_t tileWidth, StencilCoeffs c);
void encodeStencilTiledCube(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                            id<MTLBuffer> out, uint32_t N, StencilCoeffs c);
void encodeStencilPlanes(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in,
                         id<MTLBuffer> out, uint32_t N, uint32_t tileWidth, StencilCoeffs c);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso(const Kernel& k) {
  const std::string shared = PMPP_METAL_SHARED_DIR;
  switch (k.kind) {
    case Kind::Basic: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "stencil_basic", {{"TILE_WIDTH", "16"}});
    case Kind::RegTile:
      return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "stencil_regtile", {{"TILE_WIDTH", std::to_string(k.tile)}});
    case Kind::Cube: return pmpp::metal::pipeline(shared + "/tiled_cube.metal", "stencil_tiled_cube");
    case Kind::Planes:
      return pmpp::metal::pipeline(shared + "/planes.metal", "stencil_planes", {{"TILE_WIDTH", std::to_string(k.tile)}});
  }
  return nil;
}
// The most threads per threadgroup this kernel's pipeline accepts (0: it did not build).
static int maxThreads(const Kernel& k) {
  id<MTLComputePipelineState> p = pso(k);
  return p ? (int)p.maxTotalThreadsPerThreadgroup : 0;
}
static const char* kLimitName = "the pipeline's maxTotalThreadsPerThreadgroup";
static pmpp::Timing launch(const Kernel& k, Buf* in, Buf* out, unsigned N) {
  id<MTLComputePipelineState> p = pso(k);
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (k.kind) {
      case Kind::Basic: encodeStencilBasic(enc, p, buffer(in), buffer(out), N, kCoeffs); break;
      case Kind::RegTile: encodeStencilRegTiled(enc, p, buffer(in), buffer(out), N, k.depth, (uint32_t)k.tile, kCoeffs); break;
      case Kind::Cube: encodeStencilTiledCube(enc, p, buffer(in), buffer(out), N, kCoeffs); break;
      case Kind::Planes: encodeStencilPlanes(enc, p, buffer(in), buffer(out), N, (uint32_t)k.tile, kCoeffs); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
// Metal compiles the tile width in at run time: 8 and 16 always, 32 where the pipeline allows it.
static std::vector<int> tiles() { return {8, 16, 32}; }
static bool tileOptional(int tile) { return tile == 32; }
#else
void launchStencilBasic(const float* in, float* out, unsigned int N, StencilCoeffs c);
void launchStencilRegTiled(const float* in, float* out, unsigned int N, unsigned int depth, StencilCoeffs c);
void launchStencilTiledCube(const float* in, float* out, unsigned int N, StencilCoeffs c);
void launchStencilPlanes(const float* in, float* out, unsigned int N, StencilCoeffs c);

static int maxThreads(const Kernel&) { return (int)pmpp::device_prop("maxThreadsPerBlock", 1024); }
static const char* kLimitName = "maxThreadsPerBlock";
static pmpp::Timing launch(const Kernel& k, Buf* in, Buf* out, unsigned N) {
  const float* i = (const float*)pmpp::raw(in);
  float* o = (float*)pmpp::raw(out);
  return pmpp::run([&](void*) {
    switch (k.kind) {   // the tile width is fixed at compile time (TILE_WIDTH)
      case Kind::Basic: launchStencilBasic(i, o, N, kCoeffs); break;
      case Kind::RegTile: launchStencilRegTiled(i, o, N, k.depth, kCoeffs); break;
      case Kind::Cube: launchStencilTiledCube(i, o, N, kCoeffs); break;
      case Kind::Planes: launchStencilPlanes(i, o, N, kCoeffs); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static std::vector<int> tiles() { return {TILE_WIDTH}; }
static bool tileOptional(int) { return false; }
#endif

// ---------------------------------------------------------------- CPU reference
// One output point in double precision, and the sum of the magnitudes of its seven
// terms (the scale of the rounding error a float sum can make).
static inline void referencePoint(const float* in, size_t N, size_t i, size_t j, size_t k, double* value, double* mag) {
  const double t[7] = {
      (double)kCoeffs.c0 * in[i * N * N + j * N + k],
      (double)kCoeffs.c1 * in[i * N * N + j * N + (k - 1)], (double)kCoeffs.c2 * in[i * N * N + j * N + (k + 1)],
      (double)kCoeffs.c3 * in[i * N * N + (j - 1) * N + k], (double)kCoeffs.c4 * in[i * N * N + (j + 1) * N + k],
      (double)kCoeffs.c5 * in[(i - 1) * N * N + j * N + k], (double)kCoeffs.c6 * in[(i + 1) * N * N + j * N + k]};
  double v = 0, m = 0;
  for (double x : t) { v += x; m += std::fabs(x); }
  *value = v; *mag = m;
}
static inline bool interior(size_t N, size_t i, size_t j, size_t k) {
  return i >= 1 && i + 1 < N && j >= 1 && j + 1 < N && k >= 1 && k + 1 < N;
}
static const double kRtol = 1e-4;   // |got - ref| <= kRtol * (1 + sum of |terms|), as pmpp::compare

// Bit pattern written after the end of `out` before each launch (a NaN no kernel produces).
static const uint32_t kGuardBits = 0x7FC0BEEFu;
static size_t guardLength(size_t N) { return N * N + 64; }   // one plane, and a little more

static std::string where(size_t N, size_t q) {
  char b[96];
  std::snprintf(b, sizeof b, "(z, y, x) = (%zu, %zu, %zu)", q / (N * N), (q / N) % N, q % N);
  return b;
}

// Explains what is wrong with `got` (N^3 values followed by the guard region), or "".
// `wrongAt(q)` tells whether point q is outside the tolerance.
template <class WrongAt, class Expected>
static std::string explain(const std::vector<float>& in, const std::vector<float>& got, size_t N,
                           WrongAt wrongAt, Expected expected) {
  const size_t n3 = N * N * N;
  size_t guardHits = 0, firstGuard = 0;
  for (size_t q = n3; q < got.size(); ++q) {
    uint32_t bits; std::memcpy(&bits, &got[q], 4);
    if (bits != kGuardBits) { if (!guardHits) firstGuard = q - n3; guardHits++; }
  }
  size_t badIn = 0, badBd = 0, unchanged = 0, first = SIZE_MAX;
  for (size_t q = 0; q < n3; ++q) {
    if (!wrongAt(q)) continue;
    if (first == SIZE_MAX) first = q;
    const size_t i = q / (N * N), j = (q / N) % N, k = q % N;
    if (interior(N, i, j, k)) { badIn++; unchanged += (got[q] == in[q]); } else badBd++;
  }
  std::string msg;
  char b[400];
  if (guardHits) {
    std::snprintf(b, sizeof b, "the kernel wrote past the end of out: %zu guard elements changed, the first %zu "
                  "element(s) after the last grid point", guardHits, firstGuard + 1);
    msg = b;
  }
  if (first == SIZE_MAX) return msg;
  const size_t interiorCount = (N - 2) * (N - 2) * (N - 2);
  std::snprintf(b, sizeof b, "%zu of %zu interior points wrong, %zu boundary points changed; first at %s "
                "(%s): got %.6g, expected %.6g", badIn, interiorCount, badBd, where(N, first).c_str(),
                interior(N, first / (N * N), (first / N) % N, first % N) ? "interior" : "boundary",
                (double)got[first], expected(first));
  if (!msg.empty()) msg += "; ";
  msg += b;
  if (badIn && unchanged == badIn) msg += ". Those points still hold the input: nothing was written there";
  if (badBd) msg += ". Boundary points must keep their input values";
  return msg;
}

// ---------------------------------------------------------------- a test problem
struct Problem {
  size_t N, n3;
  std::vector<float> in, ref, mag, init;
  Buf *In = nullptr, *Out = nullptr;
  Problem(unsigned N_, uint32_t seed) : N(N_), n3((size_t)N_ * N_ * N_) {
    in = pmpp::random_floats(n3, seed);
    ref = in;                                        // the boundary keeps its input value
    mag.resize(n3);
    for (size_t q = 0; q < n3; ++q) mag[q] = std::fabs(in[q]);
    for (size_t i = 1; i + 1 < N; ++i)
      for (size_t j = 1; j + 1 < N; ++j)
        for (size_t k = 1; k + 1 < N; ++k) {
          double v, m;
          referencePoint(in.data(), N, i, j, k, &v, &m);
          ref[i * N * N + j * N + k] = (float)v;
          mag[i * N * N + j * N + k] = (float)m;
        }
    init.resize(n3 + guardLength(N));
    std::memcpy(init.data(), in.data(), n3 * 4);
    for (size_t q = n3; q < init.size(); ++q) std::memcpy(&init[q], &kGuardBits, 4);
    In = pmpp::alloc(n3 * 4);
    Out = pmpp::alloc(init.size() * 4);
    pmpp::upload(In, in.data(), n3 * 4);
  }
  Problem(const Problem&) = delete;
  ~Problem() { pmpp::release(In); pmpp::release(Out); }

  std::string run(const Kernel& k) {
    pmpp::upload(Out, init.data(), init.size() * 4);   // out = in, then the guard region
    launch(k, In, Out, (unsigned)N);
    std::vector<float> got(init.size());
    pmpp::download(Out, got.data(), got.size() * 4);
    pmpp::Cmp c = pmpp::compare(got.data(), ref.data(), mag.data(), n3, kRtol);
    bool guardOk = std::memcmp(got.data() + n3, init.data() + n3, (got.size() - n3) * 4) == 0;
    if (c.ok && guardOk) return "";
    return explain(in, got, N,
                   [&](size_t q) { return !(std::fabs((double)got[q] - ref[q]) <= kRtol * (1.0 + mag[q])); },
                   [&](size_t q) { return (double)ref[q]; });
  }
};

static std::map<unsigned, std::unique_ptr<Problem>> g_problems;
static Problem& problem(unsigned N) {
  auto& p = g_problems[N];
  if (!p) p.reset(new Problem(N, 800 + N));
  return *p;
}

static std::string label(const Kernel& k) {
  char b[96];
  switch (k.kind) {
    case Kind::Basic: return "basic (8x8x8 blocks)";
    case Kind::RegTile: std::snprintf(b, sizeof b, "regtile T=%d", k.tile); return b;
    case Kind::Cube: return "provided cube (Fig. 8.8, T=8)";
    case Kind::Planes: std::snprintf(b, sizeof b, "provided planes (Fig. 8.10, T=%d)", k.tile); return b;
  }
  return "?";
}

// "" if this machine can launch the kernel, else why not.
static std::string launchable(const Kernel& k) {
  const int most = maxThreads(k), need = threadsNeeded(k);
  if (most == 0) return "the kernel did not build (see the compiler message above)";
  if (most < need) {
    char b[200];
    std::snprintf(b, sizeof b, "needs %d threads per block; %s is %d", need, kLimitName, most);
    return b;
  }
  return "";
}

static void info(const pmpp::Args& a, const std::string& text) {
  std::printf("  INFO  %s\n", text.c_str());
  pmpp::Json j;
  j.s("kind", "info").s("lab", a.lab).s("variant", a.variant).s("backend", pmpp::backend()).s("text", text);
  j.emit();
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("weights: c0 %.2f | x-1 c1 %.2f, x+1 c2 %.2f | y-1 c3 %.2f, y+1 c4 %.2f | z-1 c5 %.2f, z+1 c6 %.2f\n",
              kCoeffs.c0, kCoeffs.c1, kCoeffs.c2, kCoeffs.c3, kCoeffs.c4, kCoeffs.c5, kCoeffs.c6);
  // The emulator runs one thread at a time under AddressSanitizer, three times per
  // case: it gets small grids, enough to have several blocks in every direction. The
  // race flavour runs a block's threads concurrently under ThreadSanitizer, which is
  // slower still: the small grids, and one with two blocks in every direction.
#if defined(PMPP_BACKEND_RACE)
  const std::vector<unsigned> sizes = {3, 4, 5, 34};
#else
  const std::vector<unsigned> sizes = pmpp::is_emu() ? std::vector<unsigned>{3, 4, 5, 33, 34, 63}
                                                     : std::vector<unsigned>{3, 34, 65, 130, 257};
#endif
  auto depthsFor = [](int tile, unsigned N) {
    std::vector<unsigned> d;
    for (unsigned x : {1u, 7u, (unsigned)(tile - 2), N})
      if (std::find(d.begin(), d.end(), x) == d.end()) d.push_back(x);
    return d;
  };
  auto runAll = [&](const Kernel& base, bool withDepths) {
    const std::string why = launchable(base);
    if (!why.empty()) {
      if (tileOptional(base.tile)) {
        info(a, label(base) + " not tested on this device: " + why);
      } else {
        t.check(label(base) + " can be launched here", [why] { return why; });
      }
      return;
    }
    for (unsigned N : sizes) {
      std::vector<unsigned> depths = withDepths ? depthsFor(base.tile, N) : std::vector<unsigned>{0};
      for (unsigned d : depths) {
        Kernel k = base;
        k.depth = d;
        std::string name = label(k) + ", N=" + std::to_string(N);
        if (withDepths) name += ", depth " + std::to_string(d);
        t.check(name, [&]() -> std::string { return problem(N).run(k); });
      }
    }
  };

  runAll(Kernel{Kind::Basic}, false);
  for (int tile : tiles()) runAll(Kernel{Kind::RegTile, tile}, true);
  runAll(Kernel{Kind::Cube, 8}, false);
  for (int tile : tiles()) runAll(Kernel{Kind::Planes, tile}, false);
  g_problems.clear();
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// The whole grid is checked after timing, point by point against the CPU, so that a
// fast wrong kernel is reported instead of timed.
static std::string verifyLarge(const std::vector<float>& in, const std::vector<float>& got, size_t N) {
  auto wrongAt = [&](size_t q) {
    const size_t i = q / (N * N), j = (q / N) % N, k = q % N;
    if (!interior(N, i, j, k)) return got[q] != in[q];
    double v, m;
    referencePoint(in.data(), N, i, j, k, &v, &m);
    return !(std::fabs((double)got[q] - v) <= kRtol * (1.0 + m));
  };
  bool any = false;
  for (size_t q = 0; q < N * N * N && !any; ++q) any = wrongAt(q);
  for (size_t q = N * N * N; q < got.size() && !any; ++q) { uint32_t b; std::memcpy(&b, &got[q], 4); any = b != kGuardBits; }
  if (!any) return "";
  return explain(in, got, N, wrongAt, [&](size_t q) {
    const size_t i = q / (N * N), j = (q / N) % N, k = q % N;
    if (!interior(N, i, j, k)) return (double)in[q];
    double v, m;
    referencePoint(in.data(), N, i, j, k, &v, &m);
    return v;
  });
}

// Best time of one kernel. A sweep can take well under a millisecond, and single runs
// that short vary a lot from one to the next, so after a few warm-up runs it repeats
// until the timed runs add up to about a quarter of a second (at least `reps`, at most 200).
static double bestTime(int reps, const std::function<pmpp::Timing()>& once) {
  const double first = pmpp::time_it(3, once).min_s;
  int n = reps;
  if (first > 0) n = (int)std::min(200.0, std::max((double)reps, 0.25 / first));
  return std::min(first, pmpp::time_it(n, once).min_s);
}

static int bench(const pmpp::Args& a) {
  // 1. Grid size: grow N until the basic kernel takes long enough to time, without
  //    letting in and out together pass 2 GiB or an eighth of the device's memory.
  const double mem = deviceMemory();
  const double budget = std::min(2048.0 * 1024 * 1024, mem > 0 ? mem / 8 : 256.0 * 1024 * 1024);
  auto fits = [&](size_t n) { return 8.0 * n * n * n + 4.0 * guardLength(n) <= budget; };
  size_t N = 64;
  for (;;) {
    Buf *I = pmpp::alloc(N * N * N * 4), *O = pmpp::alloc((N * N * N + guardLength(N)) * 4);
    double s = pmpp::time_it(a.reps, [&] { return launch(Kernel{Kind::Basic}, I, O, (unsigned)N); }).min_s;
    pmpp::release(I); pmpp::release(O);
    if (s >= 0.02 || !fits(2 * N)) break;
    N *= 2;
  }
  const size_t n3 = N * N * N;
  const double points = (double)(N - 2) * (N - 2) * (N - 2);   // interior points computed per sweep

  std::vector<float> in = pmpp::random_floats(n3, 2026), init(n3 + guardLength(N)), got(init.size());
  std::memcpy(init.data(), in.data(), n3 * 4);
  for (size_t q = n3; q < init.size(); ++q) std::memcpy(&init[q], &kGuardBits, 4);
  Buf *In = pmpp::alloc(n3 * 4), *Out = pmpp::alloc(init.size() * 4);
  pmpp::upload(In, in.data(), n3 * 4);

  std::printf("grid %zu x %zu x %zu: %.1f M interior points; in and out together %.0f MB\n"
              "  each kernel: best of at least %d runs, repeated until the runs add up to about 0.25 s\n",
              N, N, N, points / 1e6, 8.0 * n3 / 1e6, a.reps);
  std::printf("  'GB/s at 8 B/point' counts one 4-byte load and one 4-byte store per interior point,\n"
              "  the traffic of the 13/8 FLOP/B ideal; it is not what the kernel asked of memory.\n");
  std::printf("  %-34s %10s %9s %19s %9s %10s\n", "kernel", "ms", "Gpts/s", "GB/s at 8 B/point", "vs basic", "vs planes");

  // Times one kernel, checks its output, prints a line. Returns seconds, or 0 if it failed.
  auto measure = [&](const Kernel& k, const std::string& name, double basic_s, double planes_s) -> double {
    pmpp::upload(Out, init.data(), init.size() * 4);
    pmpp::take_error();
    const double s = bestTime(a.reps, [&] { return launch(k, In, Out, (unsigned)N); });
    std::string err = pmpp::take_error();
    if (err.empty() && s <= 0) err = "no time was recorded";
    if (err.empty()) {
      pmpp::download(Out, got.data(), got.size() * 4);
      err = verifyLarge(in, got, N);
      if (!err.empty()) err = "WRONG RESULT, not reported: " + err;
    }
    if (!err.empty()) { std::printf("  %-34s failed: %s\n", name.c_str(), err.c_str()); return 0; }
    std::printf("  %-34s %10.3f %9.2f %19.1f", name.c_str(), s * 1e3, points / s / 1e9, 8.0 * points / s / 1e9);
    if (basic_s > 0) std::printf(" %8.2fx", basic_s / s); else std::printf(" %9s", "");
    if (planes_s > 0) std::printf(" %9.2fx", planes_s / s);
    std::printf("\n");
    return s;
  };
  auto row = [&](const char* name, const Kernel& k, double s) {
    pmpp::Json j = pmpp::bench_row(a, name);
    if (k.kind != Kind::Basic) j.n("tile", k.tile);
    if (k.kind == Kind::RegTile || k.kind == Kind::Planes) j.n("depth", k.depth);
    if (k.kind == Kind::RegTile) j.n("slabs", (double)((N + k.depth - 1) / k.depth));
    j.n("n", (double)N).n("seconds", s).n("gpts", points / s / 1e9).n("gbps_compulsory", 8.0 * points / s / 1e9);
    return j;
  };

  const Kernel basic{Kind::Basic};
  const double basic_s = launchable(basic).empty() ? measure(basic, "basic, 8x8x8 blocks", 0, 0) : 0;
  if (basic_s <= 0) {
    std::printf("the basic kernel did not run correctly; nothing else is timed.\n");
    pmpp::release(In); pmpp::release(Out);
    return 1;
  }
  row("basic", basic, basic_s).emit();

  const Kernel cube{Kind::Cube, 8};
  // The cube kernel's own limit: on Metal it belongs to its compiled pipeline (section 8.4 reads it).
  std::printf("  cube: a block of %d threads; the limit (%s) is %d\n", threadsNeeded(cube), kLimitName, maxThreads(cube));
  pmpp::bench_row(a, "limits").s("kernel", "cube").n("tile", 8).n("max_threads", maxThreads(cube)).emit();
  if (launchable(cube).empty()) {
    const double s = measure(cube, "cube T=8 (provided)", basic_s, 0);
    if (s > 0) row("cube", cube, s).n("speedup_vs_basic", basic_s / s).emit();
  }

  for (int tile : tiles()) {
    const Kernel probe{Kind::RegTile, tile, (unsigned)(tile - 2)};
    std::printf("  T=%d: a block of %d threads; the limit (%s) is %d\n", tile, tile * tile, kLimitName, maxThreads(probe));
    pmpp::bench_row(a, "limits").s("kernel", "regtile").n("tile", tile).n("max_threads", maxThreads(probe)).emit();
    const std::string why = launchable(probe);
    if (!why.empty()) { std::printf("  T=%d skipped: %s\n", tile, why.c_str()); continue; }

    const Kernel planes{Kind::Planes, tile, (unsigned)(tile - 2)};
    double planes_s = 0;
    if (launchable(planes).empty()) {
      planes_s = measure(planes, "planes T=" + std::to_string(tile) + " (provided)", basic_s, 0);
      if (planes_s > 0) row("planes", planes, planes_s).n("speedup_vs_basic", basic_s / planes_s).emit();
    }
    // The book's slab (T - 2 planes), three fixed depths, and one slab for the whole
    // column. The grid has ceil(N / depth) slabs and slab 0 starts at plane 0, so depth
    // N is the only one that gives a single slab (any depth from N - 2 to N - 1 adds a
    // second slab that has at most one plane to compute and idles for the rest).
    std::vector<unsigned> depths;
    for (unsigned d : {(unsigned)(tile - 2), 8u, 16u, 64u, (unsigned)N})
      if (d <= N && std::find(depths.begin(), depths.end(), d) == depths.end()) depths.push_back(d);
    for (unsigned d : depths) {
      const Kernel k{Kind::RegTile, tile, d};
      std::string name = "regtile T=" + std::to_string(tile) + ", depth " + std::to_string(d);
      if (d == (unsigned)(tile - 2)) name += " (book)";
      if (d == N) name += " (1 slab)";
      const double s = measure(k, name, basic_s, planes_s);
      if (s <= 0) continue;
      pmpp::Json j = row("regtile", k, s);
      j.n("speedup_vs_basic", basic_s / s);
      if (planes_s > 0) j.n("speedup_vs_planes", planes_s / s);
      j.emit();
    }
  }
  pmpp::release(In); pmpp::release(Out);
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
