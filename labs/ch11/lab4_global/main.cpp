// Lab 11.4 driver: tests and benchmarks your two hierarchical global scans,
// scan-scan-add (Fig. 11.14) and reduce-scan-scan (Fig. 11.15), against Fig. 11.1's
// loop, and times them beside a copy kernel (provided, in ../shared-g13/). You do not
// need to edit this file; your work goes in work/.
#include <cstring>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The designs, in the order the lesson presents them. `key` is the "kernel" field of the
// benchmark rows that section 11.9 and the labs page read.
enum Design { ScanScanAdd, ReduceScanScan, Copy, kDesigns };
struct DesignInfo { const char* key; const char* label; };
static const DesignInfo kInfo[kDesigns] = {
    {"scan_scan_add", "scan-scan-add (Fig. 11.14)"},
    {"reduce_scan_scan", "reduce-scan-scan (Fig. 11.15)"},
    {"copy", "CopyKernel (provided)"},
};
static const int kGuard = 16;   // guard words after the output and after the scratch

// ---- entry points implemented in <variant>/ and ../shared-g13/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
int coarseFactor();
uint32_t scratchFloats(uint32_t n, uint32_t block);
bool encodeScanScanAdd(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> scanSegments,
                       id<MTLComputePipelineState> addBlockSums,
                       id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> scratch, uint32_t n, uint32_t block);
bool encodeReduceScanScan(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> reduceSegments,
                          id<MTLComputePipelineState> scanWithCarry,
                          id<MTLBuffer> in, id<MTLBuffer> out, id<MTLBuffer> scratch, uint32_t n, uint32_t block);
void encodeCopy(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>,
                id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n, uint32_t block);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
// PMPP_METAL_SOURCE is <lab>/<variant>/kernel.metal; the provided files are two levels up.
static const std::string kShared = dirOf(dirOf(PMPP_METAL_SOURCE)) + "/../shared-g13/";

enum Pipe { PScanSegments, PAddBlockSums, PReduceSegments, PScanWithCarry, PCopy };
static const char* kPipeName[] = {"scan_segments", "add_block_sums", "reduce_segments", "scan_with_carry",
                                  "copy_kernel"};
static bool segmentBuffer(int k) { return k == PScanSegments || k == PScanWithCarry; }
// COARSE_FACTOR is a macro in the Metal source, set here from launch.mm's coarseFactor().
static id<MTLComputePipelineState> pipe(int k) {
  if (k == PCopy) return pmpp::metal::pipeline(kShared + "global_copy.metal", "copy_kernel");
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, kPipeName[k],
                               {{"COARSE_FACTOR", std::to_string(coarseFactor())}});
}
static std::vector<int> pipesOf(int d) {
  if (d == ScanScanAdd) return {PScanSegments, PAddBlockSums};
  if (d == ReduceScanScan) return {PReduceSegments, PScanWithCarry};
  return {PCopy};
}

// The driver's own copy of the rule a correct launch.mm applies, used to choose which
// block sizes to test: the block fits every pipeline of the design, the scans and the
// reduction get whole SIMD-groups (at most `width` of them), and the threadgroup memory
// fits the device.
static std::string blockProblem(int d, int block) {
  char b[240];
  for (int k : pipesOf(d)) {
    id<MTLComputePipelineState> p = pipe(k);
    if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
    if ((NSUInteger)block > p.maxTotalThreadsPerThreadgroup) {
      std::snprintf(b, sizeof b, "%s runs at most %lu threads per threadgroup", kPipeName[k],
                    (unsigned long)p.maxTotalThreadsPerThreadgroup);
      return b;
    }
    if (k == PCopy) continue;
    const NSUInteger w = p.threadExecutionWidth;
    if (block % w != 0 || (NSUInteger)block / w > w) {
      std::snprintf(b, sizeof b, "SIMD-group width %lu: %d threads is not 1 to %lu whole SIMD-groups",
                    (unsigned long)w, block, (unsigned long)w);
      return b;
    }
    const NSUInteger need = p.staticThreadgroupMemoryLength +
                            (segmentBuffer(k) ? (NSUInteger)coarseFactor() * block * 4 : 0);
    if (need > pmpp::metal::device().maxThreadgroupMemoryLength) {
      std::snprintf(b, sizeof b, "%s needs %lu bytes of threadgroup memory; this device allows %lu",
                    kPipeName[k], (unsigned long)need,
                    (unsigned long)pmpp::metal::device().maxThreadgroupMemoryLength);
      return b;
    }
  }
  return "";
}

// Empty string when the design ran; otherwise why it did not.
static std::string runDesign(int d, Buf* in, Buf* out, Buf* scratch, uint32_t n, uint32_t block,
                             pmpp::Timing* tm = nullptr) {
  for (int k : pipesOf(d))
    if (!pipe(k)) return "the Metal kernel did not build: " + pmpp::take_error();
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (d) {
      case ScanScanAdd:
        ok = encodeScanScanAdd(enc, pipe(PScanSegments), pipe(PAddBlockSums), buffer(in), buffer(out),
                               buffer(scratch), n, block);
        break;
      case ReduceScanScan:
        ok = encodeReduceScanScan(enc, pipe(PReduceSegments), pipe(PScanWithCarry), buffer(in), buffer(out),
                                  buffer(scratch), n, block);
        break;
      default: encodeCopy(enc, pipe(PCopy), buffer(in), buffer(out), n, block); break;
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused block " + std::to_string(block) + ", which these pipelines can run";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
int coarseFactor();
unsigned int scratchFloats(unsigned int n, unsigned int block);
void launchScanScanAdd(const float* in, float* out, float* scratch, unsigned int n, unsigned int block);
void launchReduceScanScan(const float* in, float* out, float* scratch, unsigned int n, unsigned int block);
void launchCopy(const float* in, float* out, unsigned int n, unsigned int block);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }

// The block fits the device, and the scan kernels' shared memory fits what a block may
// use without opting in (sharedMemPerBlock; ch05-facts A3): the segment buffer plus
// threadSums (one float per thread of the largest block) plus the warp totals.
static std::string blockProblem(int d, int block) {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (block > most) return "this device runs at most " + std::to_string(most) + " threads per block";
  if (d == Copy) return "";
  const double limit = pmpp::device_prop("sharedMemPerBlock", 48 * 1024);
  const double need = 1024 * 4 + 32 * 4 + (double)coarseFactor() * block * 4;
  if (need > limit) {
    char b[200];
    std::snprintf(b, sizeof b, "needs %.0f bytes of shared memory; a block may use %.0f without opting in", need, limit);
    return b;
  }
  return "";
}
static std::string runDesign(int d, Buf* in, Buf* out, Buf* scratch, unsigned int n, unsigned int block,
                             pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    switch (d) {
      case ScanScanAdd: launchScanScanAdd(f(in), f(out), f(scratch), n, block); break;
      case ReduceScanScan: launchReduceScanScan(f(in), f(out), f(scratch), n, block); break;
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

// Integer inputs whose sum of |x| is at most 2^24 have exact float prefix sums in any
// order of addition: every partial sum is an integer float represents exactly. Those
// must match exactly. Anything else (uniform data, or integers past that size) is
// compared within 1e-5 times the sum of |x| up to the element, plus 1e-6. A global scan
// reaches each output through at most about 16 additions per level of the hierarchy
// (the thread's run, two warp scans, three carries), and the tests run up to 4 levels,
// so the worst-case float error is below 64 * 2^-24 = 3.8e-6 times that sum. Metal does
// not default to strict IEEE (ch05-facts C7), and the margin covers it.
static bool exactData(Data d, const std::vector<float>& x) {
  if (d == Uniform) return false;
  double s = 0;
  for (float v : x) s += std::fabs((double)v);
  return s <= 16777216.0;
}

// Compares got with Fig. 11.1's loop run in double, computed on the fly. Returns the
// first wrong index, or -1. *want receives the expected value there.
static long firstWrong(const std::vector<float>& x, const float* got, bool exact, double* want) {
  double acc = 0, mag = 0;
  for (size_t i = 0; i < x.size(); ++i) {
    acc += x[i];
    mag += std::fabs((double)x[i]);
    const bool ok = exact ? (double)got[i] == acc : std::fabs((double)got[i] - acc) <= 1e-5 * mag + 1e-6;
    if (!ok) { *want = acc; return (long)i; }   // also catches NaN
  }
  return -1;
}

static const float kNaN = std::numeric_limits<float>::quiet_NaN();
// The guard words: a bit pattern no scan of these inputs produces.
static float sentinel() { uint32_t u = 0x7F7FDEADu; float v; std::memcpy(&v, &u, 4); return v; }
static bool isSentinel(float v) { uint32_t u; std::memcpy(&u, &v, 4); return u == 0x7F7FDEADu; }

// Design d on n elements of kind `kind` with blocks of `block`, launched twice on the
// same buffers: first with the output and the scratch full of NaN, then with new data
// and whatever the first launch left there. Empty string = both right.
static std::string checkOne(int d, unsigned int n, unsigned int block, Data kind) {
  const unsigned int sf = d == Copy ? 0 : scratchFloats(n, block);
  const unsigned int segment = (unsigned int)coarseFactor() * block;
  Buf* in = pmpp::alloc((size_t)n * 4);
  Buf* out = pmpp::alloc(((size_t)n + kGuard) * 4);
  Buf* scratch = pmpp::alloc(((size_t)sf + kGuard) * 4);
  std::vector<float> got((size_t)n + kGuard, sentinel()), sc((size_t)sf + kGuard, sentinel()), back(n);
  std::fill(got.begin(), got.begin() + n, kNaN);            // a slot never written shows
  std::fill(sc.begin(), sc.begin() + sf, kNaN);             // a total read before it is written shows
  pmpp::upload(out, got.data(), got.size() * 4);
  pmpp::upload(scratch, sc.data(), sc.size() * 4);
  std::string err;
  for (int run = 1; run <= 2 && err.empty(); ++run) {
    std::vector<float> x = makeData(kind, n, 1000u + (uint32_t)(n * 7 + block) + 977u * (uint32_t)run);
    pmpp::upload(in, x.data(), (size_t)n * 4);
    char what[200];
    std::snprintf(what, sizeof what, "n %u, %s, run %d of 2 (%s): ", n, dataName(kind), run,
                  run == 1 ? "output and scratch start as NaN" : "new data, scratch holds run 1's values");
    err = runDesign(d, in, out, scratch, n, block);
    if (!err.empty()) { err = what + err; break; }
    pmpp::download(out, got.data(), got.size() * 4);
    pmpp::download(scratch, sc.data(), sc.size() * 4);
    pmpp::download(in, back.data(), (size_t)n * 4);
    for (size_t j = n; j < got.size() && err.empty(); ++j)
      if (!isSentinel(got[j]))
        err = std::string(what) + "out[" + std::to_string(j) + "], past the last element, was written. "
              "A thread past n must not store.";
    for (size_t j = sf; j < sc.size() && err.empty(); ++j)
      if (!isSentinel(sc[j]))
        err = std::string(what) + "scratch[" + std::to_string(j) + "] was written, but scratchFloats(" +
              std::to_string(n) + ", " + std::to_string(block) + ") is " + std::to_string(sf) +
              ": it is too small for what the launcher uses.";
    if (err.empty() && std::memcmp(back.data(), x.data(), (size_t)n * 4) != 0)
      err = std::string(what) + "the input was changed. Only the output and the scratch are yours to write.";
    if (!err.empty()) break;
    long bad = -1;
    double want = 0;
    if (d == Copy) {
      for (size_t i = 0; i < n && bad < 0; ++i)
        if (!(got[i] == x[i])) { bad = (long)i; want = x[i]; }
    } else {
      bad = firstWrong(x, got.data(), exactData(kind, x), &want);
    }
    if (bad >= 0) {
      char b[300];
      std::snprintf(b, sizeof b, "out[%ld] (segment %lu, position %lu of %u) is %.9g, expected %.9g", bad,
                    (unsigned long)(bad / segment), (unsigned long)(bad % segment), segment, got[bad], want);
      err = std::string(what) + b;
      if (std::isnan(got[bad])) err += " (NaN: nothing was written there, or a total was read before it was written)";
    }
  }
  pmpp::release(in); pmpp::release(out); pmpp::release(scratch);
  return err;
}

static std::vector<unsigned int> pick(const std::vector<unsigned int>& race, const std::vector<unsigned int>& emu,
                                      const std::vector<unsigned int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::vector<Data> data = {Ones, SmallInts, Uniform};
  // Sizes around one segment (S = coarseFactor() * block, 256 at block 64) and past one
  // segment of segments. On the emulator, n = 70001 at block 64 is 274 segments, whose
  // 274 totals make 2 segments, whose 2 totals make 1: three levels of the recursion.
  // The emulator is slow at that size, so it runs it for the scans only, at block 64,
  // with the integers only (exact, and the strictest of the three kinds). On `race`, n = 5000 at block 64 is 20 blocks, several of them at
  // once.
  const std::vector<unsigned int> ns =
      pick({1000, 5000}, {1, 2, 255, 256, 257, 1000, 4097, 70001},
           {1, 2, 255, 256, 257, 1000, 4096, 4097, 65536, 70001, (1u << 20) + 1, (1u << 24) + 7});
  const std::vector<unsigned int> blocks = pick({64}, {64, 256}, {64, 256, 1024});
  const unsigned int emuAllKindsUpTo = pmpp::is_emu() ? 5000u : ~0u;
  auto skip = [&](int d, unsigned int n, unsigned int block, Data k) {
    return n > emuAllKindsUpTo && (d == Copy || k != SmallInts || block != 64);
  };
  std::printf("COARSE_FACTOR %d: a block of B threads owns a segment of %d*B elements\n", coarseFactor(),
              coarseFactor());
  for (int d = 0; d < kDesigns; ++d) {
    for (unsigned int block : blocks) {
      const std::string name = std::string(kInfo[d].label) + ", block " + std::to_string(block);
      std::string why = blockProblem(d, (int)block);
      if (!why.empty() && why.rfind("the Metal kernel did not build", 0) != 0) {
        std::printf("  skip  %s: %s\n", name.c_str(), why.c_str());
        continue;
      }
      t.check(name, [&]() -> std::string {
        if (!why.empty()) return why;
        for (unsigned int n : ns)
          for (Data k : data) {
            if (skip(d, n, block, k)) continue;
            std::string err = checkOne(d, n, block, k);
            if (!err.empty()) return err;
          }
        return "";
      });
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Best of `reps` timed runs after one untimed warm-up. pmpp::run times everything the
// launcher enqueues: every kernel of every level (on Metal, the GPU's own start and end
// times of the command buffer; on CUDA and HIP, host time around launch and synchronise).
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

// The book's global-memory traffic per element (pp. 274-275), S elements per segment.
static double bookBytes(int d, double S) {
  return d == ScanScanAdd ? 16.0 + 8.0 / S : d == ReduceScanScan ? 12.0 + 12.0 / S : 8.0;
}

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes the input, output and scratch may take
  unsigned int n = 1u << 26;
  while (8.0 * n * 1.01 > budget && n > (1u << 16)) n /= 2;
  std::vector<float> x = pmpp::random_floats(n, 21);
  unsigned int sfMax = 0;
  for (unsigned int block : {256u, 1024u}) sfMax = std::max(sfMax, scratchFloats(n, block));
  Buf* in = pmpp::alloc((size_t)n * 4);
  Buf* out = pmpp::alloc((size_t)n * 4);
  Buf* scratch = pmpp::alloc((size_t)sfMax * 4);
  pmpp::upload(in, x.data(), (size_t)n * 4);
  int rc = 0;

  // Keep the GPU busy for a quarter of a second first (copies whose results are thrown
  // away), so the first kernel timed does not also pay for the device waking up.
  {
    double busy = 0;
    for (int i = 0; i < 10000 && busy < 0.25; ++i) {
      pmpp::Timing tm{0, 0};
      if (!runDesign(Copy, in, out, scratch, n, 256, &tm).empty()) break;
      busy += tm.wall_s;
    }
    pmpp::take_error();
  }

  std::printf("n = %u floats (%.0f MB in, %.0f MB out), COARSE_FACTOR %d, best of %d runs\n", n, n * 4 / 1e6,
              n * 4 / 1e6, coarseFactor(), a.reps);
  std::printf("  %-6s %-6s %-30s %9s %9s %12s %11s %8s\n", "block", "S", "design", "time ms", "Gelem/s",
              "book B/elem", "GB/s book", "of copy");
  std::vector<float> got(n);
  for (unsigned int block : {256u, 1024u}) {
    const unsigned int S = (unsigned int)coarseFactor() * block;
    double copy_s = 0, seconds[kDesigns] = {0, 0, 0};
    for (int d : {Copy, ScanScanAdd, ReduceScanScan}) {
      std::string why = blockProblem(d, (int)block);
      if (!why.empty()) { std::printf("  %-6u %-6u %-30s skipped: %s\n", block, S, kInfo[d].label, why.c_str()); continue; }
      std::string err;
      double s = bestOf(a.reps, [&](pmpp::Timing* tm) { return runDesign(d, in, out, scratch, n, block, tm); }, &err);
      if (s <= 0) { std::printf("  %-6u %-6u %-30s failed: %s\n", block, S, kInfo[d].label, err.c_str()); rc = 1; continue; }
      pmpp::download(out, got.data(), (size_t)n * 4);
      double want = 0;
      long bad = -1;
      if (d == Copy) {
        if (std::memcmp(got.data(), x.data(), (size_t)n * 4) != 0) bad = 0;
      } else {
        bad = firstWrong(x, got.data(), false, &want);
      }
      if (bad >= 0) {
        std::printf("  %-6u %-6u %-30s wrong result (first at %ld); not reported\n", block, S, kInfo[d].label, bad);
        rc = 1;
        continue;
      }
      if (d == Copy) copy_s = s;
      seconds[d] = s;
      const double B = bookBytes(d, S);
      const double gbps = B * n / s / 1e9;
      const double share = copy_s > 0 ? copy_s / s : 0.0;
      std::printf("  %-6u %-6u %-30s %9.3f %9.2f %12.3f %11.1f %7.0f%%\n", block, S, kInfo[d].label, s * 1e3,
                  n / s / 1e9, B, gbps, share * 100);
      pmpp::bench_row(a, "global_scan").s("kernel", kInfo[d].key).n("block", block).n("segment", S)
          .n("coarse", coarseFactor()).n("n", n).n("seconds", s).n("elements_per_s", n / s)
          .n("book_bytes_per_element", B).n("gb_per_s", gbps).n("share_of_copy", share).emit();
    }
    if (seconds[ScanScanAdd] > 0 && seconds[ReduceScanScan] > 0)
      std::printf("  %-6u %-6u reduce-scan-scan ran %.2f times as fast as scan-scan-add; the byte count says %.3f\n",
                  block, S, seconds[ScanScanAdd] / seconds[ReduceScanScan],
                  bookBytes(ScanScanAdd, S) / bookBytes(ReduceScanScan, S));
  }
  std::printf("  Gelem/s is n divided by the time. \"book B/elem\" is the book's count of global-memory bytes\n"
              "  per element (pp. 272-275: 8 for a copy, 16 + 8/S, 12 + 12/S), and \"GB/s book\" is that count\n"
              "  times n, divided by the time. \"of copy\" is the copy's time as a share of the design's.\n"
              "\nThese are measurements of this machine, at this size.\n");
  pmpp::release(in); pmpp::release(out); pmpp::release(scratch);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
