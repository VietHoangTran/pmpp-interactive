// Lab 11.2 driver: tests and benchmarks your coarsened and register-tiled block scans
// against a provided thread-major variant (each thread loads its own run directly from
// global memory) and a copy kernel, both in ../shared-g11/. You do not need to edit
// this file; your work goes in work/.
#include <cstring>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The kernels, in the order the lesson presents them. `key` is the "kernel" field of the
// benchmark rows that sections 11.6, 11.7 and 11.8 and the labs page read.
enum Kind { Coarsened, RegisterTiled, ThreadMajor, Copy, kKinds };
struct KernelInfo { const char* key; const char* label; bool segmentInShared; };
static const KernelInfo kInfo[kKinds] = {
    {"coarsened", "CoarsenedScanKernel", true},
    {"register_tiled", "RegisterTiledScanKernel", true},
    {"thread_major", "ThreadMajorScanKernel (provided)", false},
    {"copy", "CopyKernel (provided)", false},
};
static const int kFactors[] = {1, 2, 4, 8, 16};
// The scan kernels' static shared arrays: threadSums (one float per thread of the
// largest block) and the warp totals (1024/32 floats). The segment buffer comes on top.
static const double kStaticSharedBytes = 1024 * 4 + 32 * 4;

// ---- entry points implemented in <variant>/ and ../shared-g11/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeCoarsenedScan(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                         id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block, uint32_t coarse);
bool encodeRegisterTiledScan(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                             id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block, uint32_t coarse);
void encodeThreadMajorScan(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block, uint32_t coarse);
void encodeCopy(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
// PMPP_METAL_SOURCE is <lab>/<variant>/kernel.metal; the provided files are two levels up.
static const std::string kShared = dirOf(dirOf(PMPP_METAL_SOURCE)) + "/../shared-g11/";

// One pipeline per kernel and factor: COARSE_FACTOR is a macro in the Metal source.
static id<MTLComputePipelineState> pipelineFor(int k, int coarse) {
  const std::vector<std::pair<std::string, std::string>> m = {{"COARSE_FACTOR", std::to_string(coarse)}};
  switch (k) {
    case Coarsened: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "coarsened_scan", m);
    case RegisterTiled: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "register_tiled_scan", m);
    case ThreadMajor: return pmpp::metal::pipeline(kShared + "bs_threadmajor.metal", "thread_major_scan", m);
    default: return pmpp::metal::pipeline(kShared + "bs_copy.metal", "copy_kernel");
  }
}

// The driver's own copy of the rule a correct launch.mm applies, used to choose which
// configurations to test: the block fits the pipeline, the scans get whole SIMD-groups
// (at most `width` of them), and the threadgroup memory fits the device.
static std::string configProblem(int k, int block, int coarse) {
  id<MTLComputePipelineState> p = pipelineFor(k, coarse);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  char b[240];
  if ((NSUInteger)block > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline runs at most %lu threads per threadgroup",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  if (k == Copy) return "";
  const NSUInteger w = p.threadExecutionWidth;
  if (block % w != 0 || (NSUInteger)block / w > w) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads is not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, block, (unsigned long)w);
    return b;
  }
  const NSUInteger need = p.staticThreadgroupMemoryLength +
                          (kInfo[k].segmentInShared ? (NSUInteger)coarse * block * 4 : 0);
  if (need > pmpp::metal::device().maxThreadgroupMemoryLength) {
    std::snprintf(b, sizeof b, "needs %lu bytes of threadgroup memory; this device allows %lu",
                  (unsigned long)need, (unsigned long)pmpp::metal::device().maxThreadgroupMemoryLength);
    return b;
  }
  return "";
}

// Empty string when the kernel ran; otherwise why it did not.
static std::string runKernel(int k, Buf* in, Buf* out, int n, int block, int coarse, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = pipelineFor(k, coarse);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t un = (uint32_t)n, ub = (uint32_t)block, uc = (uint32_t)coarse;
    switch (k) {
      case Coarsened: ok = encodeCoarsenedScan(enc, p, buffer(in), buffer(out), un, ub, uc); break;
      case RegisterTiled: ok = encodeRegisterTiledScan(enc, p, buffer(in), buffer(out), un, ub, uc); break;
      case ThreadMajor: encodeThreadMajorScan(enc, p, buffer(in), buffer(out), un, ub, uc); break;
      default: encodeCopy(enc, p, buffer(in), buffer(out), un, ub); break;
    }
  });
  if (tm) *tm = t;
  if (!ok)
    return "your launch.mm refused block " + std::to_string(block) + " at factor " + std::to_string(coarse) +
           ", which this pipeline and device can run";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchCoarsenedScan(const float* in, float* out, int n, int block, int coarse);
void launchRegisterTiledScan(const float* in, float* out, int n, int block, int coarse);
void launchThreadMajorScan(const float* in, float* out, int n, int block, int coarse);
void launchCopy(const float* in, float* out, int n, int block);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }

// The block fits the device, and the shared memory fits what a block may use without
// opting in (sharedMemPerBlock; ch05-facts A3).
static std::string configProblem(int k, int block, int coarse) {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (block > most) return "this device runs at most " + std::to_string(most) + " threads per block";
  if (k == Copy) return "";
  const double limit = pmpp::device_prop("sharedMemPerBlock", 48 * 1024);
  const double need = kStaticSharedBytes + (kInfo[k].segmentInShared ? (double)coarse * block * 4 : 0);
  if (need > limit) {
    char b[200];
    std::snprintf(b, sizeof b, "needs %.0f bytes of shared memory; a block may use %.0f without opting in", need, limit);
    return b;
  }
  return "";
}
static std::string runKernel(int k, Buf* in, Buf* out, int n, int block, int coarse, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    switch (k) {
      case Coarsened: launchCoarsenedScan(f(in), f(out), n, block, coarse); break;
      case RegisterTiled: launchRegisterTiledScan(f(in), f(out), n, block, coarse); break;
      case ThreadMajor: launchThreadMajorScan(f(in), f(out), n, block, coarse); break;
      default: launchCopy(f(in), f(out), n, block); break;
    }
  });
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
// Ones and small integers have exact float prefix sums here (every one is an integer of
// magnitude at most 3*8192), whatever the order of the additions, so they must match
// exactly. Uniform data is compared within 1e-6 times the sum of |x| over the segment up
// to that element, plus 1e-6: the kernels add in a different order from the loop, and
// Metal does not default to strict IEEE.
static bool matches(Data d, float got, double ref, double mag) {
  if (d != Uniform) return (double)got == ref;
  return std::fabs((double)got - ref) <= 1e-6 * mag + 1e-6;   // also false for NaN
}
// Fig. 11.1's loop in double, restarted at every segment of `seg` elements. Calls
// visit(i, ref, mag) for each element in order, where mag is the sum of |x| over the
// segment up to i. Streams, so the benchmark's large arrays need no reference copy.
template <typename Visit>
static void refScan(const std::vector<float>& x, size_t seg, Visit visit) {
  const size_t n = x.size();
  for (size_t s = 0; s < n; s += seg) {
    double acc = 0, a = 0;
    for (size_t i = s; i < n && i < s + seg; ++i) {
      acc += x[i];
      a += std::fabs((double)x[i]);
      visit(i, acc, a);
    }
  }
}
static const float kNaN = std::numeric_limits<float>::quiet_NaN();
static float sentinel() { uint32_t u = 0x7F7FDEADu; float v; std::memcpy(&v, &u, 4); return v; }
static bool isSentinel(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u == 0x7F7FDEADu; }

static std::string describe(size_t i, size_t seg, float got, double want, const char* what) {
  char b[400];
  std::snprintf(b, sizeof b, "out[%zu] (%s, segment %zu, position %zu) is %.9g, expected %.9g", i, what,
                i / seg, i % seg, got, want);
  std::string s = b;
  if (std::isnan(got)) s += " (NaN: nothing was written there)";
  return s;
}

// One launch of kernel k on n elements of kind d. Empty string = right everywhere and
// nothing written past n.
static std::string checkOne(int k, int n, int block, int coarse, Data d) {
  const size_t seg = k == Copy ? (size_t)block : (size_t)coarse * block;
  std::vector<float> x = makeData(d, (size_t)n, 3000u + (uint32_t)(n * 7 + block * 3 + coarse));
  // Guard words up to the end of the last segment, and 16 more.
  const size_t guard = ((size_t)n + seg - 1) / seg * seg - n + 16;
  Buf* in = pmpp::alloc((size_t)n * 4);
  Buf* out = pmpp::alloc(((size_t)n + guard) * 4);
  pmpp::upload(in, x.data(), (size_t)n * 4);
  std::vector<float> got((size_t)n + guard, sentinel());
  std::fill(got.begin(), got.begin() + n, kNaN);          // a slot never written shows
  pmpp::upload(out, got.data(), got.size() * 4);
  std::string err = runKernel(k, in, out, n, block, coarse);
  const std::string what = "n " + std::to_string(n) + ", " + dataName(d) + ": ";
  if (err.empty()) {
    pmpp::download(out, got.data(), got.size() * 4);
    for (size_t j = n; j < got.size() && err.empty(); ++j)
      if (!isSentinel(got[j]))
        err = what + "out[" + std::to_string(j) + "], past the last element, was written. "
              "Nothing at or past n may be stored.";
    if (k == Copy) {
      for (size_t i = 0; i < (size_t)n && err.empty(); ++i)
        if (!(got[i] == x[i])) err = what + describe(i, seg, got[i], x[i], "copy");
    } else if (err.empty()) {
      std::vector<double> ref((size_t)n), mag((size_t)n);
      refScan(x, seg, [&](size_t i, double r, double m) { ref[i] = r; mag[i] = m; });
      // 1. Every segment restarts: its first element is its own input, and its last
      //    element (the end of the segment, or n - 1 in a partial last segment) is the
      //    segment's total.
      for (size_t s = 0; s < (size_t)n && err.empty(); s += seg) {
        if (!matches(d, got[s], ref[s], mag[s])) {
          err = what + describe(s, seg, got[s], ref[s], "first of its segment");
          if (s > 0 && !std::isnan(got[s])) err += ". Each block scans only its own segment, starting from 0.";
        }
        const size_t last = std::min((size_t)n, s + seg) - 1;
        const bool partial = last + 1 == (size_t)n && (size_t)n % seg != 0;
        if (err.empty() && !matches(d, got[last], ref[last], mag[last]))
          err = what + describe(last, seg, got[last], ref[last],
                                partial ? "last element, in the partial last segment" : "last of its segment");
      }
      // 2. Every element.
      for (size_t i = 0; i < (size_t)n && err.empty(); ++i)
        if (!matches(d, got[i], ref[i], mag[i])) err = what + describe(i, seg, got[i], ref[i], "inside");
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
  const std::vector<int> ns = pick({1000}, {1, 2, 33, 1000, 4097},
                                   {1, 2, 33, 1000, 4097, 65537, (1 << 22) + 3});
  const std::vector<int> blocks = pick({64}, {64, 256}, {64, 256, 1024});
  const std::vector<int> factors = pick({1, 4, 8}, {1, 4, 8}, {1, 2, 4, 8, 16});
  auto one = [&](int k, int block, int coarse, const std::string& name) {
    std::string why = configProblem(k, block, coarse);
    if (!why.empty() && why.rfind("the Metal kernel did not build", 0) != 0) {
      std::printf("  skip  %s: %s\n", name.c_str(), why.c_str());
      return;
    }
    t.check(name, [&]() -> std::string {
      if (!why.empty()) return why;
      for (int n : ns)
        for (Data d : data) {
          std::string err = checkOne(k, n, block, coarse, d);
          if (!err.empty()) return err;
        }
      return "";
    });
  };
  for (int k : {Coarsened, RegisterTiled, ThreadMajor})
    for (int coarse : factors)
      for (int block : blocks)
        one(k, block, coarse, std::string(kInfo[k].label) + ", factor " + std::to_string(coarse) +
                                  ", block " + std::to_string(block));
  for (int block : blocks) one(Copy, block, 1, std::string(kInfo[Copy].label) + ", block " + std::to_string(block));
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

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes the input and output may take
  size_t n = (size_t)1 << 26;
  while (8.0 * n > budget && n > ((size_t)1 << 16)) n /= 2;
  std::vector<float> x = pmpp::random_floats(n, 31);
  Buf* in = pmpp::alloc(n * 4);
  Buf* out = pmpp::alloc(n * 4);
  pmpp::upload(in, x.data(), n * 4);
  std::vector<float> got(n);
  int rc = 0;

  // Keep the GPU busy for a quarter of a second first (copies whose results are thrown
  // away), so the first kernel timed does not also pay for the device waking up.
  {
    double busy = 0;
    for (int i = 0; i < 10000 && busy < 0.25; ++i) {
      pmpp::Timing tm{0, 0};
      if (!runKernel(Copy, in, out, (int)n, 256, 1, &tm).empty()) break;
      busy += tm.wall_s;
    }
    pmpp::take_error();
  }

  std::printf("n = %zu floats (%.0f MB in, %.0f MB out), one segment of factor*block elements per block,"
              " best of %d runs\n", n, n * 4 / 1e6, n * 4 / 1e6, a.reps);
  std::printf("  %-6s %-7s %-34s %10s %10s %9s\n", "block", "factor", "kernel", "time ms", "GB/s", "of copy");
  for (int block : {256, 1024}) {
    std::string err;
    double copy_s = 0;
    std::string why = configProblem(Copy, block, 1);
    if (why.empty()) copy_s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runKernel(Copy, in, out, (int)n, block, 1, tm); }, &err);
    if (copy_s > 0) {
      pmpp::download(out, got.data(), n * 4);
      if (std::memcmp(got.data(), x.data(), n * 4) != 0) { std::printf("  copy at block %d gave wrong data\n", block); rc = 1; copy_s = 0; }
    }
    if (copy_s <= 0) {
      std::printf("  %-6d %-7s %-34s %s\n", block, "-", kInfo[Copy].label, why.empty() ? ("failed: " + err).c_str() : ("skipped: " + why).c_str());
      continue;
    }
    const double copy_gbps = 8.0 * n / copy_s / 1e9;
    std::printf("  %-6d %-7s %-34s %10.4f %10.1f %8.0f%%\n", block, "-", kInfo[Copy].label, copy_s * 1e3, copy_gbps, 100.0);
    pmpp::bench_row(a, "scan").s("kernel", kInfo[Copy].key).n("block", block).n("n", (double)n)
        .n("seconds", copy_s).n("gb_per_s", copy_gbps).n("share_of_copy", 1.0).emit();
    for (int coarse : kFactors) {
      for (int k : {Coarsened, RegisterTiled, ThreadMajor}) {
        why = configProblem(k, block, coarse);
        if (!why.empty()) { std::printf("  %-6d %-7d %-34s skipped: %s\n", block, coarse, kInfo[k].label, why.c_str()); continue; }
        double s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runKernel(k, in, out, (int)n, block, coarse, tm); }, &err);
        if (s <= 0) { std::printf("  %-6d %-7d %-34s failed: %s\n", block, coarse, kInfo[k].label, err.c_str()); rc = 1; continue; }
        pmpp::download(out, got.data(), n * 4);
        size_t bad = 0;
        refScan(x, (size_t)coarse * block, [&](size_t i, double r, double m) { bad += !matches(Uniform, got[i], r, m); });
        if (bad) { std::printf("  %-6d %-7d %-34s %zu of %zu elements wrong; not reported\n", block, coarse, kInfo[k].label, bad, n); rc = 1; continue; }
        const double gbps = 8.0 * n / s / 1e9;
        std::printf("  %-6d %-7d %-34s %10.4f %10.1f %8.0f%%\n", block, coarse, kInfo[k].label, s * 1e3, gbps, copy_s / s * 100);
        pmpp::bench_row(a, "scan").s("kernel", kInfo[k].key).n("block", block).n("coarse", coarse).n("n", (double)n)
            .n("seconds", s).n("gb_per_s", gbps).n("share_of_copy", copy_s / s).emit();
      }
    }
  }
  std::printf("  GB/s counts 8 bytes per element (one 4-byte read, one 4-byte write), divided by the time.\n"
              "  \"of copy\" is the kernel's rate as a share of the copy kernel's at the same block size.\n"
              "\nThese are measurements of this machine, at this size.\n");
  pmpp::release(in); pmpp::release(out);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
