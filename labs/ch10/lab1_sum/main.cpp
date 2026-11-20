// Lab 10.1 driver: tests and benchmarks three sum reductions (yours: shared-memory,
// two-stage warp, coarsened with an atomic commit) against the book's first two
// (provided: simple and convergent, in ../shared/). You do not need to edit this file;
// your work goes in work/.
#include <cstring>
#include <limits>
#include <set>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The row kernels, in the order the lesson presents them. `key` is the "kernel" field of
// the benchmark rows that sections 10.8 to 10.10 read.
struct RowKernel { const char* key; const char* label; bool provided; bool inPlace; };
static const RowKernel kRows[4] = {
    {"simple", "SimpleSumReductionKernel (provided)", true, true},
    {"convergent", "ConvergentSumReductionKernel (provided)", true, true},
    {"shared", "SharedMemorySumReductionKernel", false, false},
    {"warp", "TwoStageWarpLevelSumReductionKernel", false, false},
};

// ---- entry points implemented in <variant>/ and ../shared/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeSharedMemorySum(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block);
bool encodeTwoStageWarpSum(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block);
bool encodeCoarsenedSum(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block, uint32_t coarse);
void encodeSimpleSum(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                     id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block);
void encodeConvergentSum(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                         id<MTLBuffer> in, id<MTLBuffer> out, uint32_t rows, uint32_t block);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> rowPipeline(int k) {
  const std::string provided = std::string(PMPP_METAL_SHARED_DIR) + "/inplace_sum.metal";
  switch (k) {
    case 0: return pmpp::metal::pipeline(provided, "simple_sum");
    case 1: return pmpp::metal::pipeline(provided, "convergent_sum");
    case 2: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "shared_memory_sum");
    default: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "two_stage_warp_sum");
  }
}
static id<MTLComputePipelineState> anyPipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "coarsened_sum"); }

// The driver's own copy of the rule a correct launch.mm applies, used to choose which
// block sizes to test. The SIMD-group kernels (warp, coarsened) need a block that is a
// whole number of SIMD-groups, and no more SIMD-groups than one SIMD-group has lanes.
static std::string blockProblem(id<MTLComputePipelineState> p, int block, bool simd) {
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  char b[200];
  if ((NSUInteger)block > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline runs at most %lu threads per threadgroup",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  const NSUInteger w = p.threadExecutionWidth;
  if (simd && (block % w != 0 || (NSUInteger)block / w > w)) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads is not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, block, (unsigned long)w);
    return b;
  }
  return "";
}
static std::string rowBlockProblem(int k, int block) { return blockProblem(rowPipeline(k), block, k == 3); }
static std::string anyBlockProblem(int block) { return blockProblem(anyPipeline(), block, true); }

// Empty string when the kernel ran; otherwise why it did not.
static std::string runRows(int k, Buf* in, Buf* out, int rows, int block, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = rowPipeline(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (k) {
      case 0: encodeSimpleSum(enc, p, buffer(in), buffer(out), rows, block); break;
      case 1: encodeConvergentSum(enc, p, buffer(in), buffer(out), rows, block); break;
      case 2: ok = encodeSharedMemorySum(enc, p, buffer(in), buffer(out), rows, block); break;
      default: ok = encodeTwoStageWarpSum(enc, p, buffer(in), buffer(out), rows, block); break;
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused block " + std::to_string(block) + ", which this pipeline can run";
  return "";
}
static std::string runAny(Buf* in, Buf* out, int n, int block, int coarse, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = anyPipeline();
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    ok = encodeCoarsenedSum(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out),
                            (uint32_t)n, (uint32_t)block, (uint32_t)coarse);
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused block " + std::to_string(block) + ", which this pipeline can run";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchSimpleSum(float* in, float* out, int rows, int block);
void launchConvergentSum(float* in, float* out, int rows, int block);
void launchSharedMemorySum(const float* in, float* out, int rows, int block);
void launchTwoStageWarpSum(const float* in, float* out, int rows, int block);
void launchCoarsenedSum(const float* in, float* out, int n, int block, int coarse);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }

static std::string threadsProblem(int block) {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return block <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block";
}
static std::string rowBlockProblem(int, int block) { return threadsProblem(block); }
static std::string anyBlockProblem(int block) { return threadsProblem(block); }

static std::string runRows(int k, Buf* in, Buf* out, int rows, int block, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    switch (k) {
      case 0: launchSimpleSum(f(in), f(out), rows, block); break;
      case 1: launchConvergentSum(f(in), f(out), rows, block); break;
      case 2: launchSharedMemorySum(f(in), f(out), rows, block); break;
      default: launchTwoStageWarpSum(f(in), f(out), rows, block); break;
    }
  });
  if (tm) *tm = t;
  return "";
}
static std::string runAny(Buf* in, Buf* out, int n, int block, int coarse, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) { launchCoarsenedSum(f(in), f(out), n, block, coarse); });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- data and reference
enum Data { Ones, SmallInts, Uniform };
static const char* dataName(Data d) {
  return d == Ones ? "all ones" : d == SmallInts ? "integers in [-3, 3]" : "uniform in [-1, 1]";
}
static std::vector<float> makeData(Data d, size_t n, uint32_t seed) {
  if (d == Ones) return std::vector<float>(n, 1.0f);
  std::vector<float> v = pmpp::random_floats(n, seed);
  if (d == SmallInts)
    for (float& x : v) x = (float)std::lround(x * 3.0f);
  return v;
}
// Sum and sum of |x| of v[lo, hi), in double.
static void refSum(const std::vector<float>& v, size_t lo, size_t hi, double* sum, double* mag) {
  double s = 0, a = 0;
  for (size_t i = lo; i < hi; ++i) { s += v[i]; a += std::fabs((double)v[i]); }
  *sum = s; *mag = a;
}
// Ones and small integers have exact float sums here (every partial sum is an integer
// below 2^24), whatever the order of the additions, so they must match exactly. Uniform
// data is compared with a tolerance relative to sum |x|, plus an absolute floor: float
// atomics on global memory flush denormals on CUDA (ch10-facts A6), so a result near 0
// must not be judged relative to itself.
static bool matches(Data d, float got, double ref, double mag) {
  if (d != Uniform) return (double)got == ref;
  return std::fabs((double)got - ref) <= 1e-6 * mag + 1e-6;   // also false for NaN
}
static std::string mismatch(const std::string& where, float got, double ref) {
  char b[300];
  std::snprintf(b, sizeof b, "%s is %.9g, expected %.9g", where.c_str(), got, ref);
  std::string s = b;
  if (std::isnan(got)) s += " (NaN: nothing was written there, or it started from NaN)";
  return s;
}
static const float kNaN = std::numeric_limits<float>::quiet_NaN();

// One launch of row kernel k over `rows` rows of 2*block floats of kind d.
// Empty string = every row correct.
static std::string checkRows(int k, int rows, int block, Data d) {
  const size_t n = (size_t)rows * 2 * block;
  std::vector<float> x = makeData(d, n, 1000u + (uint32_t)(rows * 7 + block));
  Buf* in = pmpp::alloc(n * 4);
  Buf* out = pmpp::alloc((size_t)rows * 4);
  pmpp::upload(in, x.data(), n * 4);
  std::vector<float> got(rows, kNaN);
  pmpp::upload(out, got.data(), (size_t)rows * 4);       // NaN: a row never written shows
  std::string err = runRows(k, in, out, rows, block);
  if (err.empty()) {
    pmpp::download(out, got.data(), (size_t)rows * 4);
    for (int r = 0; r < rows && err.empty(); ++r) {
      double sum, mag;
      refSum(x, (size_t)r * 2 * block, (size_t)(r + 1) * 2 * block, &sum, &mag);
      if (!matches(d, got[r], sum, mag))
        err = std::to_string(rows) + (rows == 1 ? " row, " : " rows, ") + dataName(d) + ": " + mismatch("out[" + std::to_string(r) + "]", got[r], sum);
    }
  }
  pmpp::release(in); pmpp::release(out);
  return err;
}

// Two launches of the coarsened kernel on the same output, which starts as NaN: the
// launcher must reset *out to 0 itself, every time.
static std::string checkAny(int n, int block, int coarse, Data d) {
  std::vector<float> x = makeData(d, (size_t)n, 2000u + (uint32_t)n);
  double sum, mag;
  refSum(x, 0, (size_t)n, &sum, &mag);
  Buf* in = pmpp::alloc((size_t)n * 4);
  Buf* out = pmpp::alloc(4);
  pmpp::upload(in, x.data(), (size_t)n * 4);
  float got = kNaN;
  pmpp::upload(out, &got, 4);
  std::string err;
  const std::string what = "n " + std::to_string(n) + ", " + dataName(d) + ": ";
  for (int launch = 1; launch <= 2 && err.empty(); ++launch) {
    err = runAny(in, out, n, block, coarse);
    if (!err.empty()) break;
    pmpp::download(out, &got, 4);
    if (!matches(d, got, sum, mag)) {
      err = what + mismatch(launch == 1 ? "the sum" : "the sum from a second launch on the same output", got, sum);
      if (launch == 2 && matches(d, got, 2 * sum, 2 * mag)) err += " (twice the sum: is *out reset to 0 before each launch?)";
      if (launch == 1 && std::isnan(got)) err += "; *out starts as NaN here, so the launcher must set it to 0";
    }
  }
  pmpp::release(in); pmpp::release(out);
  return err;
}

static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<Data> data = {Ones, SmallInts, Uniform};

  // Forms (a) and (b), and the provided baselines (so a broken provided file shows).
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 3, 37});
  const std::vector<int> rowBlocks = pick({64}, {64, 256}, {64, 128, 256, 1024});
  for (int k = 0; k < 4; ++k) {
    for (int block : rowBlocks) {
      const std::string name = std::string(kRows[k].label) + ", block " + std::to_string(block);
      std::string why = rowBlockProblem(k, block);
      if (!why.empty() && why.rfind("the Metal kernel did not build", 0) != 0) {
        std::printf("  skip  %s: %s\n", name.c_str(), why.c_str());
        continue;
      }
      t.check(name, [&]() -> std::string {
        if (!why.empty()) return why;
        for (int rows : rowCounts)
          for (Data d : data) {
            std::string err = checkRows(k, rows, block, d);
            if (!err.empty()) return err;
          }
        return "";
      });
    }
  }

  // Form (c): any length, any coarsening factor, an atomic commit per block. On `race`
  // the sizes keep at least 4 blocks in most cases, so that blocks run concurrently and
  // a plain (non-atomic) commit to *out is reported.
  const std::vector<int> ns = pick({33, 1000, 4097}, {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097},
                                   {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003, 1 << 22});
  const std::vector<int> anyBlocks = pick({64}, {64, 256}, {64, 256});
  for (int block : anyBlocks) {
    for (int coarse : {1, 2, 3, 8}) {
      const std::string name = "CoarsenedSumReductionKernel, block " + std::to_string(block) +
                               ", coarse " + std::to_string(coarse);
      std::string why = anyBlockProblem(block);
      if (!why.empty() && why.rfind("the Metal kernel did not build", 0) != 0) {
        std::printf("  skip  %s: %s\n", name.c_str(), why.c_str());
        continue;
      }
      t.check(name, [&]() -> std::string {
        if (!why.empty()) return why;
        for (int n : ns)
          for (Data d : data) {
            std::string err = checkAny(n, block, coarse, d);
            if (!err.empty()) return err;
          }
        return "";
      });
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Best of `reps` timed runs after one untimed warm-up. pmpp::run times only what the
// launcher enqueues (on Metal, the GPU's own start and end times of the command buffer).
static double bestOf(int reps, const std::function<std::string(pmpp::Timing*)>& once, std::string* err) {
  double best = 1e30;
  for (int r = -1; r < reps; ++r) {
    pmpp::Timing tm{0, 0};
    std::string e = once(&tm);
    if (e.empty()) e = pmpp::take_error();
    if (!e.empty()) { *err = e; return -1; }
    if (r >= 0) best = std::min(best, tm.best());
  }
  return best;
}

// The largest power of two from 64 to `want` that every kernel in the list can run.
static int benchBlock(int want, bool rowsToo) {
  for (int b = want; b >= 64; b /= 2) {
    bool ok = anyBlockProblem(b).empty();
    if (rowsToo) for (int k = 0; k < 4; ++k) ok = ok && rowBlockProblem(k, b).empty();
    if (ok) return b;
  }
  return 0;
}

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one input array may take
  int rc = 0;

  // (1) Rows: the four row kernels on the same 2^24 floats.
  {
    const int block = benchBlock(256, true);
    if (!block) { std::printf("rows: no block size from 64 to 256 runs every kernel here\n"); return 1; }
    // The provided kernels overwrite their input, so every run, warm-up included, gets
    // its own copy, uploaded before any timing starts. The runs then follow each other
    // with nothing in between, as they do for the other kernels.
    const int copies = a.reps + 1;
    size_t total = (size_t)1 << 24;
    while ((double)total * 4 * copies > budget && total > ((size_t)1 << 16)) total /= 2;
    const int rows = (int)(total / (2 * (size_t)block));
    std::vector<float> x = pmpp::random_floats(total, 11);
    std::vector<double> sum(rows), mag(rows);
    for (int r = 0; r < rows; ++r) refSum(x, (size_t)r * 2 * block, (size_t)(r + 1) * 2 * block, &sum[r], &mag[r]);
    std::vector<Buf*> in(copies);
    for (Buf*& b : in) b = pmpp::alloc(total * 4);
    Buf* out = pmpp::alloc((size_t)rows * 4);
    // Keep the GPU busy for a quarter of a second first (the provided convergent kernel,
    // whose results are thrown away), so the first kernel timed does not also pay for
    // the device waking up.
    {
      double busy = 0;
      for (int i = 0; i < 10000 && busy < 0.25; ++i) {
        pmpp::Timing tm{0, 0};
        if (!runRows(1, in[0], out, rows, block, &tm).empty()) break;
        busy += tm.wall_s;
      }
      pmpp::take_error();
    }
    std::printf("rows: %zu floats (%.0f MB) as %d rows of %d (block %d), best of %d runs\n",
                total, total * 4 / 1e6, rows, 2 * block, block, a.reps);
    std::printf("  %-42s %10s %10s %10s\n", "kernel", "time ms", "GB/s", "vs simple");
    double simple_s = 0;
    for (int k = 0; k < 4; ++k) {
      for (Buf* b : in) pmpp::upload(b, x.data(), total * 4);
      std::string err;
      int run = 0;
      double s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runRows(k, in[run++], out, rows, block, tm); }, &err);
      if (s <= 0) { std::printf("  %-42s failed: %s\n", kRows[k].label, err.c_str()); rc = 1; continue; }
      std::vector<float> got(rows);
      pmpp::download(out, got.data(), (size_t)rows * 4);
      int bad = 0;
      for (int r = 0; r < rows; ++r) bad += !matches(Uniform, got[r], sum[r], mag[r]);
      if (bad) { std::printf("  %-42s %d of %d rows wrong; not reported\n", kRows[k].label, bad, rows); rc = 1; continue; }
      if (k == 0) simple_s = s;
      const double gbps = total * 4.0 / s / 1e9;
      std::printf("  %-42s %10.4f %10.1f %9.2fx\n", kRows[k].label, s * 1e3, gbps, simple_s > 0 ? simple_s / s : 0.0);
      pmpp::bench_row(a, "rows").s("kernel", kRows[k].key).n("block", block).n("rows", rows)
          .n("seconds", s).n("gb_per_s", gbps).emit();
    }
    std::printf("  GB/s counts the bytes of input read once (4 per float), divided by the time.\n");
    for (Buf* b : in) pmpp::release(b);
    pmpp::release(out);
  }

  // (2) Any length: the coarsened kernel from coarse 1 to 128 on one large array.
  const int block = benchBlock(256, false);
  if (!block) { std::printf("any: no block size from 64 to 256 runs the coarsened kernel here\n"); return 1; }
  {
    const size_t cap = std::min((size_t)1 << 28, (size_t)(budget / 4));
    size_t n = (size_t)1 << 26;
    while (n > cap && n > ((size_t)1 << 20)) n /= 2;
    std::vector<float> x;
    Buf *in = nullptr, *out = pmpp::alloc(4);
    double s = 0;
    std::string err;
    // Start at 2^26 and double while coarse 1 takes under half a millisecond (too short
    // to time well), up to 2^28 floats and a quarter of the device's memory.
    for (;;) {
      x = pmpp::random_floats(n, 12);
      if (in) pmpp::release(in);
      in = pmpp::alloc(n * 4);
      pmpp::upload(in, x.data(), n * 4);
      s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runAny(in, out, (int)n, block, 1, tm); }, &err);
      if (s <= 0 || s >= 0.0005 || n * 2 > cap) break;
      n *= 2;
    }
    if (s <= 0) {
      std::printf("any: the coarsened kernel failed: %s\n", err.c_str());
      pmpp::release(in); pmpp::release(out);
      return 1;
    }
    double sum, mag;
    refSum(x, 0, n, &sum, &mag);
    std::printf("\nany length: n = %zu floats (%.0f MB), block %d, best of %d runs\n", n, n * 4 / 1e6, block, a.reps);
    std::printf("  %-8s %10s %10s %10s\n", "coarse", "blocks", "time ms", "GB/s");
    for (int coarse = 1; coarse <= 128; coarse *= 2) {
      s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runAny(in, out, (int)n, block, coarse, tm); }, &err);
      const size_t blocks = (n + 2 * (size_t)coarse * block - 1) / (2 * (size_t)coarse * block);
      if (s <= 0) { std::printf("  %-8d failed: %s\n", coarse, err.c_str()); rc = 1; continue; }
      float got;
      pmpp::download(out, &got, 4);
      if (!matches(Uniform, got, sum, mag)) {
        std::printf("  %-8d %10zu  %s; not reported\n", coarse, blocks, mismatch("the sum", got, sum).c_str());
        rc = 1;
        continue;
      }
      const double gbps = n * 4.0 / s / 1e9;
      std::printf("  %-8d %10zu %10.4f %10.1f\n", coarse, blocks, s * 1e3, gbps);
      pmpp::bench_row(a, "any").n("n", (double)n).n("block", block).n("coarse", coarse)
          .n("blocks", (double)blocks).n("seconds", s).n("gb_per_s", gbps).emit();
    }
#if defined(PMPP_BACKEND_METAL)
    std::printf("  The time is the GPU's. Your launch.mm zeroes *out from the CPU before the dispatch, outside it.\n");
#else
    std::printf("  The time includes the launcher's cudaMemset of one float.\n");
#endif
    pmpp::release(in); pmpp::release(out);
  }

  // (3) Determinism: the same input 20 times through the atomic version.
  {
    size_t n = (size_t)1 << 24;
    while ((double)n * 4 > budget && n > ((size_t)1 << 16)) n /= 2;
    std::vector<float> x = pmpp::random_floats(n, 13);
    Buf* in = pmpp::alloc(n * 4);
    Buf* out = pmpp::alloc(4);
    pmpp::upload(in, x.data(), n * 4);
    const int runs = 20;
    std::set<uint32_t> bits;
    float lo = 0, hi = 0;
    std::string err;
    for (int r = 0; r < runs && err.empty(); ++r) {
      err = runAny(in, out, (int)n, block, 1);
      if (err.empty()) err = pmpp::take_error();
      float got;
      pmpp::download(out, &got, 4);
      uint32_t u;
      std::memcpy(&u, &got, 4);
      bits.insert(u);
      lo = r == 0 ? got : std::min(lo, got);
      hi = r == 0 ? got : std::max(hi, got);
    }
    if (!err.empty()) {
      std::printf("determinism: failed: %s\n", err.c_str());
      rc = 1;
    } else {
      const size_t blocks = (n + 2 * (size_t)block - 1) / (2 * (size_t)block);
      std::printf("\ndeterminism: %d runs of the coarsened kernel (coarse 1, block %d, %zu blocks) on the same %zu floats\n",
                  runs, block, blocks, n);
      std::printf("  %zu distinct result%s (bit patterns); smallest %.9g, largest %.9g\n",
                  bits.size(), bits.size() == 1 ? "" : "s", lo, hi);
      pmpp::bench_row(a, "determinism").n("n", (double)n).n("runs", runs).n("distinct", (double)bits.size()).emit();
    }
    pmpp::release(in); pmpp::release(out);
  }
  std::printf("\nThese are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
