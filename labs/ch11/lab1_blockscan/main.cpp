// Lab 11.1 driver: tests and benchmarks your block scans (double-buffered Kogge-Stone,
// warp-shuffle scan, and on Metal the simd_prefix_inclusive_sum scan) against the
// book's two-barrier kernel and a copy kernel (both provided, in ../shared-g11/). You
// do not need to edit this file; your work goes in work/.
#include <cstring>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The kernels, in the order the lesson presents them. `key` is the "kernel" field of the
// benchmark rows that sections 11.3 and 11.4 and the labs page read.
enum Kind { TwoBarrier, DoubleBuffer, WarpShuffle, WarpPrefix, Copy, kKinds };
struct KernelInfo { const char* key; const char* label; bool simd; bool metalOnly; };
static const KernelInfo kInfo[kKinds] = {
    {"two_barrier", "KoggeStoneKernel, two barriers (provided)", false, false},
    {"double_buffer", "KoggeStoneDoubleBufferKernel", false, false},
    {"warp_shuffle", "WarpScanKernel (shuffle loop)", true, false},
    {"warp_prefix", "warp_scan_prefix (simd_prefix_inclusive_sum)", true, true},
    {"copy", "CopyKernel (provided)", false, false},
};

// ---- entry points implemented in <variant>/ and ../shared-g11/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeKoggeStoneDB(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                        id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
bool encodeWarpScanShuffle(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                           id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
bool encodeWarpScanPrefix(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                          id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
void encodeKoggeStone(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                      id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
void encodeCopy(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
// PMPP_METAL_SOURCE is <lab>/<variant>/kernel.metal; the provided files are two levels up.
static const std::string kShared = dirOf(dirOf(PMPP_METAL_SOURCE)) + "/../shared-g11/";

static id<MTLComputePipelineState> pipelineFor(int k) {
  switch (k) {
    case TwoBarrier: return pmpp::metal::pipeline(kShared + "bs_two_barrier.metal", "ks_two_barrier");
    case DoubleBuffer: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "ks_double_buffer");
    case WarpShuffle: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "warp_scan_shuffle");
    case WarpPrefix: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "warp_scan_prefix");
    default: return pmpp::metal::pipeline(kShared + "bs_copy.metal", "copy_kernel");
  }
}
static bool available(int) { return true; }

// The driver's own copy of the rule a correct launch.mm applies, used to choose which
// block sizes to test. The SIMD-group kernels need a block that is a whole number of
// SIMD-groups, and no more SIMD-groups than one SIMD-group has lanes.
static std::string blockProblem(int k, int block) {
  id<MTLComputePipelineState> p = pipelineFor(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  char b[200];
  if ((NSUInteger)block > p.maxTotalThreadsPerThreadgroup) {
    std::snprintf(b, sizeof b, "this pipeline runs at most %lu threads per threadgroup",
                  (unsigned long)p.maxTotalThreadsPerThreadgroup);
    return b;
  }
  const NSUInteger w = p.threadExecutionWidth;
  if (kInfo[k].simd && (block % w != 0 || (NSUInteger)block / w > w)) {
    std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads is not 1 to %lu whole SIMD-groups",
                  (unsigned long)w, block, (unsigned long)w);
    return b;
  }
  return "";
}

// Empty string when the kernel ran; otherwise why it did not.
static std::string runKernel(int k, Buf* in, Buf* out, int n, int block, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = pipelineFor(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t un = (uint32_t)n, ub = (uint32_t)block;
    switch (k) {
      case TwoBarrier: encodeKoggeStone(enc, p, buffer(in), buffer(out), un, ub); break;
      case DoubleBuffer: ok = encodeKoggeStoneDB(enc, p, buffer(in), buffer(out), un, ub); break;
      case WarpShuffle: ok = encodeWarpScanShuffle(enc, p, buffer(in), buffer(out), un, ub); break;
      case WarpPrefix: ok = encodeWarpScanPrefix(enc, p, buffer(in), buffer(out), un, ub); break;
      default: encodeCopy(enc, p, buffer(in), buffer(out), un, ub); break;
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused block " + std::to_string(block) + ", which this pipeline can run";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchKoggeStone(const float* in, float* out, int n, int block);
void launchKoggeStoneDB(const float* in, float* out, int n, int block);
void launchWarpScan(const float* in, float* out, int n, int block);
void launchCopy(const float* in, float* out, int n, int block);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }

static bool available(int k) { return !kInfo[k].metalOnly; }
static std::string blockProblem(int, int block) {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return block <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block";
}
static std::string runKernel(int k, Buf* in, Buf* out, int n, int block, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    switch (k) {
      case TwoBarrier: launchKoggeStone(f(in), f(out), n, block); break;
      case DoubleBuffer: launchKoggeStoneDB(f(in), f(out), n, block); break;
      case WarpShuffle: launchWarpScan(f(in), f(out), n, block); break;
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
// Fig. 11.1's loop in double, restarted at every segment of `seg` elements. mag[i] is
// the sum of |x| over the segment up to and including i, which scales the tolerance.
static void refScan(const std::vector<float>& x, size_t seg, std::vector<double>& ref, std::vector<double>& mag) {
  const size_t n = x.size();
  ref.assign(n, 0.0);
  mag.assign(n, 0.0);
  for (size_t s = 0; s < n; s += seg) {
    double acc = 0, a = 0;
    for (size_t i = s; i < n && i < s + seg; ++i) {
      acc += x[i];
      a += std::fabs((double)x[i]);
      ref[i] = acc;
      mag[i] = a;
    }
  }
}
// Ones and small integers have exact float prefix sums here (every one is an integer of
// magnitude at most 3*1024), whatever the order of the additions, so they must match
// exactly. Uniform data is compared within 1e-6 times sum |x| plus 1e-6: a tree adds in
// a different order from the loop, and Metal does not default to strict IEEE.
static bool matches(Data d, float got, double ref, double mag) {
  if (d != Uniform) return (double)got == ref;
  return std::fabs((double)got - ref) <= 1e-6 * mag + 1e-6;   // also false for NaN
}
static const float kNaN = std::numeric_limits<float>::quiet_NaN();
// The guard words after the output: a bit pattern no scan of these inputs produces.
static float sentinel() { uint32_t u = 0x7F7FDEADu; float v; std::memcpy(&v, &u, 4); return v; }
static bool isSentinel(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u == 0x7F7FDEADu; }

// One launch of kernel k on n elements of kind d with blocks of `block`.
// Empty string = every element right and nothing written past n.
static std::string checkOne(int k, int n, int block, Data d) {
  std::vector<float> x = makeData(d, (size_t)n, 1000u + (uint32_t)(n * 7 + block));
  // Guard words up to the end of the last block, and 16 more.
  const size_t guard = ((size_t)n + block - 1) / block * block - n + 16;
  Buf* in = pmpp::alloc((size_t)n * 4);
  Buf* out = pmpp::alloc(((size_t)n + guard) * 4);
  pmpp::upload(in, x.data(), (size_t)n * 4);
  std::vector<float> got((size_t)n + guard, sentinel());
  std::fill(got.begin(), got.begin() + n, kNaN);          // a slot never written shows
  pmpp::upload(out, got.data(), got.size() * 4);
  std::string err = runKernel(k, in, out, n, block);
  const std::string what = "n " + std::to_string(n) + ", " + dataName(d) + ": ";
  if (err.empty()) {
    pmpp::download(out, got.data(), got.size() * 4);
    for (size_t j = n; j < got.size() && err.empty(); ++j)
      if (!isSentinel(got[j]))
        err = what + "out[" + std::to_string(j) + "], past the last element, was written. "
              "A thread past n must not store.";
    std::vector<double> ref, mag;
    if (k != Copy) refScan(x, (size_t)block, ref, mag);
    for (size_t i = 0; i < (size_t)n && err.empty(); ++i) {
      const double want = k == Copy ? (double)x[i] : ref[i];
      const bool ok = k == Copy ? (got[i] == x[i]) : matches(d, got[i], ref[i], mag[i]);
      if (ok) continue;
      char b[300];
      std::snprintf(b, sizeof b, "out[%zu] (segment %zu, position %zu) is %.9g, expected %.9g", i,
                    i / block, i % block, got[i], want);
      err = what + b;
      if (std::isnan(got[i])) err += " (NaN: nothing was written there)";
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
  // On `race`, n = 33 and 1000 at block 64 are 1 and 16 blocks, so several blocks run
  // at once and a missing barrier is reported with its source lines.
  const std::vector<int> ns = pick({33, 1000}, {1, 31, 32, 33, 255, 256, 257, 1000, 4097},
                                   {1, 31, 32, 33, 255, 256, 257, 1000, 4097, 100003, 1 << 22});
  const std::vector<int> blocks = pick({64}, {64, 256}, {64, 128, 256, 1024});
  for (int k = 0; k < kKinds; ++k) {
    if (!available(k)) continue;
    for (int block : blocks) {
      const std::string name = std::string(kInfo[k].label) + ", block " + std::to_string(block);
      std::string why = blockProblem(k, block);
      if (!why.empty() && why.rfind("the Metal kernel did not build", 0) != 0) {
        std::printf("  skip  %s: %s\n", name.c_str(), why.c_str());
        continue;
      }
      t.check(name, [&]() -> std::string {
        if (!why.empty()) return why;
        for (int n : ns)
          for (Data d : data) {
            std::string err = checkOne(k, n, block, d);
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

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes the input and output may take
  size_t n = (size_t)1 << 24;
  while (8.0 * n > budget && n > ((size_t)1 << 16)) n /= 2;
  std::vector<float> x = pmpp::random_floats(n, 21);
  Buf* in = pmpp::alloc(n * 4);
  Buf* out = pmpp::alloc(n * 4);
  pmpp::upload(in, x.data(), n * 4);
  int rc = 0;

  // Keep the GPU busy for a quarter of a second first (copies whose results are thrown
  // away), so the first kernel timed does not also pay for the device waking up.
  {
    double busy = 0;
    for (int i = 0; i < 10000 && busy < 0.25; ++i) {
      pmpp::Timing tm{0, 0};
      if (!runKernel(Copy, in, out, (int)n, 256, &tm).empty()) break;
      busy += tm.wall_s;
    }
    pmpp::take_error();
  }

  std::printf("n = %zu floats (%.0f MB in, %.0f MB out), one segment per block, best of %d runs\n",
              n, n * 4 / 1e6, n * 4 / 1e6, a.reps);
  std::printf("  %-6s %-46s %10s %10s %9s\n", "block", "kernel", "time ms", "GB/s", "of copy");
  const int order[kKinds] = {Copy, TwoBarrier, DoubleBuffer, WarpShuffle, WarpPrefix};
  std::vector<float> got(n);
  for (int block : {256, 512, 1024}) {
    std::vector<double> ref, mag;
    refScan(x, (size_t)block, ref, mag);
    double copy_s = 0;
    for (int k : order) {
      if (!available(k)) continue;
      std::string why = blockProblem(k, block);
      if (!why.empty()) { std::printf("  %-6d %-46s skipped: %s\n", block, kInfo[k].label, why.c_str()); continue; }
      std::string err;
      double s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runKernel(k, in, out, (int)n, block, tm); }, &err);
      if (s <= 0) { std::printf("  %-6d %-46s failed: %s\n", block, kInfo[k].label, err.c_str()); rc = 1; continue; }
      pmpp::download(out, got.data(), n * 4);
      size_t bad = 0;
      for (size_t i = 0; i < n; ++i)
        bad += k == Copy ? !(got[i] == x[i]) : !matches(Uniform, got[i], ref[i], mag[i]);
      if (bad) { std::printf("  %-6d %-46s %zu of %zu elements wrong; not reported\n", block, kInfo[k].label, bad, n); rc = 1; continue; }
      if (k == Copy) copy_s = s;
      const double gbps = 8.0 * n / s / 1e9;
      const double share = copy_s > 0 ? copy_s / s : 0.0;
      std::printf("  %-6d %-46s %10.4f %10.1f %8.0f%%\n", block, kInfo[k].label, s * 1e3, gbps, share * 100);
      pmpp::bench_row(a, "scan").s("kernel", kInfo[k].key).n("block", block).n("n", (double)n)
          .n("seconds", s).n("gb_per_s", gbps).n("share_of_copy", share).emit();
    }
  }
  std::printf("  GB/s counts 8 bytes per element (one 4-byte read, one 4-byte write), divided by the time.\n"
              "  \"of copy\" is the kernel's rate as a share of the copy kernel's at the same block size.\n"
              "\nThese are measurements of this machine, at this size.\n");
  pmpp::release(in); pmpp::release(out);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
