// Lab 12.3 driver: tests your two stable filters (part A: keep flags, scan, scatter; part B:
// count, scan of the counts, staged scatter) against a CPU stable filter, tests the provided
// scan on its own, and benchmarks both parts against a plain copy. You do not need to edit
// this file; your work goes in work/.
#include <cstring>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static const uint32_t kBlock = 256;              // BLOCK_DIM in kernel.cu, kBlock in launch.mm
static const uint32_t kThreshold = 0x80000000u;  // keep a key when key < kThreshold
static const uint32_t kSentinel = 0xDEADBEEFu;   // fills the output before a run; never a kept key
static const uint32_t kGarbage = 0xCDCDCDCDu;    // fills the scratch arrays before a run
static const uint32_t kPad = 64;                 // sentinel slots past the end of the output
static const std::vector<int> kCoarse = {1, 2, 4, 8, 16};

// ---- entry points implemented in <variant>/ and ../shared-g12/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeStableFilterFlags(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> keepFlags,
                             id<MTLComputePipelineState> scatterFlags, id<MTLBuffer> input, id<MTLBuffer> output,
                             uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize, id<MTLBuffer> keep,
                             id<MTLBuffer> offset, id<MTLBuffer> scanScratch);
bool encodeStableFilterStaged(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> countKept,
                              id<MTLComputePipelineState> stagedScatter, id<MTLBuffer> input, id<MTLBuffer> output,
                              uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize, id<MTLBuffer> blockCounts,
                              id<MTLBuffer> blockOffsets, id<MTLBuffer> scanScratch, uint32_t coarse);
uint32_t scanScratchElems(uint32_t n);
bool encodeExclusiveScanU32(id<MTLComputeCommandEncoder>, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n,
                            id<MTLBuffer> scratch);
bool encodeCopyU32(id<MTLComputeCommandEncoder>, id<MTLBuffer> in, id<MTLBuffer> out, uint32_t n);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso(const char* name) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, name); }
static id<MTLComputeCommandEncoder> encoderOf(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }

// Why part B at this coarsening factor cannot run on this device, or "". The same rule a correct
// launch.mm applies: 256 threads in whole SIMD-groups, and the staging buffer (256 * coarse keys)
// plus one slot per SIMD-group within the device's threadgroup memory.
static std::string stagedProblem(int coarse) {
  id<MTLComputePipelineState> c = pso("count_kept"), s = pso("staged_scatter");
  if (!c || !s) return "a part B Metal kernel did not build: " + pmpp::take_error();
  const NSUInteger w = c.threadExecutionWidth;
  if (kBlock > c.maxTotalThreadsPerThreadgroup || kBlock > s.maxTotalThreadsPerThreadgroup || w == 0 || kBlock % w)
    return "these pipelines cannot run 256 threads per threadgroup in whole SIMD-groups";
  const double need = (double)s.staticThreadgroupMemoryLength + kBlock * coarse * 4.0 + ((kBlock / w * 4 + 15) / 16 * 16);
  const double have = pmpp::device_prop("maxThreadgroupMemoryLength", 0);
  if (need > have) {
    char b[160];
    std::snprintf(b, sizeof b, "needs %.0f bytes of threadgroup memory, the device offers %.0f", need, have);
    return b;
  }
  return "";
}
static std::string flagsProblem() {
  id<MTLComputePipelineState> k = pso("keep_flags"), s = pso("scatter_flags");
  if (!k || !s) return "a part A Metal kernel did not build: " + pmpp::take_error();
  if (kBlock > k.maxTotalThreadsPerThreadgroup || kBlock > s.maxTotalThreadsPerThreadgroup)
    return "these pipelines cannot run 256 threads per threadgroup";
  return "";
}

static std::string runFlags(Buf* in, Buf* out, uint32_t n, Buf* size, Buf* keep, Buf* offset, Buf* scratch,
                            pmpp::Timing* tm = nullptr) {
  std::string why = flagsProblem();
  if (!why.empty()) return why;
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    ok = encodeStableFilterFlags(encoderOf(v), pso("keep_flags"), pso("scatter_flags"), buffer(in), buffer(out), n,
                                 kThreshold, buffer(size), buffer(keep), buffer(offset), buffer(scratch));
  });
  if (tm) *tm = t;
  return ok ? "" : "encodeStableFilterFlags returned false, although these pipelines can run 256 threads per threadgroup";
}
static std::string runStaged(Buf* in, Buf* out, uint32_t n, Buf* size, Buf* counts, Buf* offsets, Buf* scratch,
                             int coarse, pmpp::Timing* tm = nullptr) {
  std::string why = stagedProblem(coarse);
  if (!why.empty()) return why;
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    ok = encodeStableFilterStaged(encoderOf(v), pso("count_kept"), pso("staged_scatter"), buffer(in), buffer(out), n,
                                  kThreshold, buffer(size), buffer(counts), buffer(offsets), buffer(scratch),
                                  (uint32_t)coarse);
  });
  if (tm) *tm = t;
  return ok ? "" : "encodeStableFilterStaged returned false, although this coarsening factor fits the device";
}
static std::string runScan(Buf* in, Buf* out, uint32_t n, Buf* scratch) {
  bool ok = true;
  pmpp::run([&](void* v) { ok = encodeExclusiveScanU32(encoderOf(v), buffer(in), buffer(out), n, buffer(scratch)); });
  return ok ? "" : "the provided scan did not run: " + pmpp::take_error();
}
static pmpp::Timing runCopy(Buf* in, Buf* out, uint32_t n) {
  return pmpp::run([&](void* v) { encodeCopyU32(encoderOf(v), buffer(in), buffer(out), n); });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchStableFilterFlags(const unsigned int* input, unsigned int* output, unsigned int n, unsigned int threshold,
                             unsigned int* outputSize, unsigned int* keep, unsigned int* offset,
                             unsigned int* scanScratch);
void launchStableFilterStaged(const unsigned int* input, unsigned int* output, unsigned int n, unsigned int threshold,
                              unsigned int* outputSize, unsigned int* blockCounts, unsigned int* blockOffsets,
                              unsigned int* scanScratch, int coarse);
unsigned int scanScratchElems(unsigned int n);
void launchExclusiveScanU32(const unsigned int* in, unsigned int* out, unsigned int n, unsigned int* scratch);
void launchCopyU32(const unsigned int* in, unsigned int* out, unsigned int n);
static unsigned int* u(Buf* b) { return (unsigned int*)pmpp::raw(b); }

static std::string stagedProblem(int coarse) {
  // Dynamic staging (256 * coarse keys) plus the static per-warp totals (8 slots).
  const double need = kBlock * coarse * 4.0 + 8 * 4.0;
  const double have = pmpp::device_prop("sharedMemPerBlock", 48 * 1024);
  if (need > have) {
    char b[160];
    std::snprintf(b, sizeof b, "needs %.0f bytes of shared memory per block, the device offers %.0f", need, have);
    return b;
  }
  return "";
}
static std::string runFlags(Buf* in, Buf* out, uint32_t n, Buf* size, Buf* keep, Buf* offset, Buf* scratch,
                            pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchStableFilterFlags(u(in), u(out), n, kThreshold, u(size), u(keep), u(offset), u(scratch));
  });
  if (tm) *tm = t;
  return "";
}
static std::string runStaged(Buf* in, Buf* out, uint32_t n, Buf* size, Buf* counts, Buf* offsets, Buf* scratch,
                             int coarse, pmpp::Timing* tm = nullptr) {
  std::string why = stagedProblem(coarse);
  if (!why.empty()) return why;
  pmpp::Timing t = pmpp::run([&](void*) {
    launchStableFilterStaged(u(in), u(out), n, kThreshold, u(size), u(counts), u(offsets), u(scratch), coarse);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runScan(Buf* in, Buf* out, uint32_t n, Buf* scratch) {
  pmpp::run([&](void*) { launchExclusiveScanU32(u(in), u(out), n, u(scratch)); });
  return "";
}
static pmpp::Timing runCopy(Buf* in, Buf* out, uint32_t n) {
  return pmpp::run([&](void*) { launchCopyU32(u(in), u(out), n); });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- test data
// Kept keys are distinct, below the threshold and not in increasing order, so any key out of
// place is caught: key(i) = (i * 2654435761) mod 2^31, a one-to-one map on [0, 2^31). Dropped
// keys are 2^31 or above.
static uint32_t keptKey(uint32_t i) { return (i * 2654435761u) & 0x7FFFFFFFu; }

enum Pattern { Random, FirstOnly, LastOnly, EveryOther, RunsAcrossTiles, OneFullTile };
static const char* patternName(Pattern p) {
  switch (p) {
    case FirstOnly: return "only the first key";
    case LastOnly: return "only the last key";
    case EveryOther: return "every other key";
    case RunsAcrossTiles: return "runs of 12 across every tile boundary";
    case OneFullTile: return "one run exactly filling one tile";
    default: return "random";
  }
}

// Whether key i is kept. `tile` is the number of keys one block owns, for the two tile patterns.
struct KeepRule {
  Pattern p;
  double fraction;
  uint32_t tile;
  uint32_t fullTile;   // which tile OneFullTile fills
};
static std::vector<uint32_t> makeKeys(uint32_t n, const KeepRule& r, uint32_t seed) {
  std::vector<uint32_t> keys(n);
  uint32_t x = seed * 2654435761u + 12345u;
  const double cut = r.fraction * 4294967296.0;
  for (uint32_t i = 0; i < n; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;   // xorshift32
    bool keep = false;
    switch (r.p) {
      case FirstOnly: keep = i == 0; break;
      case LastOnly: keep = i == n - 1; break;
      case EveryOther: keep = i % 2 == 0; break;
      case RunsAcrossTiles: keep = (i + 5) % r.tile < 12; break;
      case OneFullTile: keep = i / r.tile == r.fullTile; break;
      default: keep = r.fraction >= 1.0 || (double)x < cut; break;
    }
    keys[i] = keep ? keptKey(i) : (0x80000000u | (x >> 1));
  }
  return keys;
}

// The CPU stable filter: the kept keys in input order, and where each came from.
static std::vector<uint32_t> cpuFilter(const std::vector<uint32_t>& keys, std::vector<uint32_t>* from = nullptr) {
  std::vector<uint32_t> out;
  if (from) from->clear();
  for (uint32_t i = 0; i < (uint32_t)keys.size(); ++i) {
    if (keys[i] < kThreshold) {
      out.push_back(keys[i]);
      if (from) from->push_back(i);
    }
  }
  return out;
}

// ---------------------------------------------------------------- device buffers for one n
struct Problem {
  uint32_t n;
  Buf *in, *out, *size, *keep, *offset, *scratch, *counts, *offsets;
  explicit Problem(uint32_t n_) : n(n_) {
    const uint32_t blocks = (n + kBlock - 1) / kBlock;   // part B's most blocks: coarse 1
    in = pmpp::alloc((size_t)n * 4);
    out = pmpp::alloc((size_t)(n + kPad) * 4);
    size = pmpp::alloc(4);
    keep = pmpp::alloc((size_t)n * 4);
    offset = pmpp::alloc((size_t)n * 4);
    scratch = pmpp::alloc((size_t)scanScratchElems(n) * 4);
    counts = pmpp::alloc((size_t)blocks * 4);
    offsets = pmpp::alloc((size_t)blocks * 4);
  }
  ~Problem() {
    for (Buf* b : {in, out, size, keep, offset, scratch, counts, offsets}) pmpp::release(b);
  }
  Problem(const Problem&) = delete;
  static void fill(Buf* b, size_t elems, uint32_t v) {
    std::vector<uint32_t> h(elems, v);
    pmpp::upload(b, h.data(), elems * 4);
  }
  // Fresh input; sentinel in the output and the count; garbage in every scratch array.
  void prepare(const std::vector<uint32_t>& keys) {
    pmpp::upload(in, keys.data(), (size_t)n * 4);
    fill(out, n + kPad, kSentinel);
    fill(size, 1, kSentinel);
    const uint32_t blocks = (n + kBlock - 1) / kBlock;
    fill(keep, n, kGarbage);
    fill(offset, n, kGarbage);
    fill(scratch, scanScratchElems(n), kGarbage);
    fill(counts, blocks, kGarbage);
    fill(offsets, blocks, kGarbage);
  }
  // Empty string when the device's output equals the CPU stable filter of `keys`.
  std::string check(const std::vector<uint32_t>& keys) {
    std::vector<uint32_t> from;
    const std::vector<uint32_t> ref = cpuFilter(keys, &from);
    uint32_t count = 0;
    pmpp::download(size, &count, 4);
    char b[300];
    if (count != ref.size()) {
      if (count == kSentinel) return "*outputSize was never written";
      std::snprintf(b, sizeof b, "*outputSize is %u, expected %zu", count, ref.size());
      return b;
    }
    std::vector<uint32_t> got(n + kPad);
    pmpp::download(out, got.data(), got.size() * 4);
    for (uint32_t i = 0; i < count; ++i) {
      if (got[i] != ref[i]) {
        uint32_t src = UINT32_MAX;
        for (uint32_t j = 0; j < n; ++j) if (keys[j] == got[i]) { src = j; break; }
        if (got[i] == kSentinel)
          std::snprintf(b, sizeof b, "output[%u] was never written; it should hold the key from input[%u]", i, from[i]);
        else if (src != UINT32_MAX)
          std::snprintf(b, sizeof b, "output[%u] holds the key from input[%u]; it should hold the key from input[%u]",
                        i, src, from[i]);
        else
          std::snprintf(b, sizeof b, "output[%u] is 0x%08x, which is no input key; it should hold the key from input[%u]",
                        i, got[i], from[i]);
        return b;
      }
    }
    for (uint32_t i = count; i < n + kPad; ++i) {
      if (got[i] != kSentinel) {
        std::snprintf(b, sizeof b, "output[%u] was written, but only output[0..%u) belongs to the result", i, count);
        return b;
      }
    }
    return "";
  }
};

// ---------------------------------------------------------------- tests
static std::string fmtN(uint32_t n) {
  if (n == (1u << 20)) return "2^20";
  if (n == 3u * (1u << 20) + 17) return "3 x 2^20 + 17";
  return std::to_string(n);
}
static const std::vector<double> kFractions = {0.0, 0.001, 0.5, 0.97, 1.0};

// One filter run on prepared buffers; `coarse` 0 means part A.
static std::string runPart(Problem& p, int coarse) {
  if (coarse == 0) return runFlags(p.in, p.out, p.n, p.size, p.keep, p.offset, p.scratch);
  return runStaged(p.in, p.out, p.n, p.size, p.counts, p.offsets, p.scratch, coarse);
}
static std::string partName(int coarse) {
  return coarse == 0 ? "part A (flags, scan, scatter)" : "part B (staged), coarse " + std::to_string(coarse);
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  const bool race = be == "race", emu = be == "emu";

  // ---- the provided scan on its own
  std::vector<uint32_t> scanSizes = race ? std::vector<uint32_t>{1, 1024, 1025, 2048}
                                  : emu ? std::vector<uint32_t>{1, 31, 1023, 1024, 1025, 4099, 65537}
                                        : std::vector<uint32_t>{1, 31, 1023, 1024, 1025, 4099, 65537, (1u << 20) + 3,
                                                                3u * (1u << 20) + 17};
  for (uint32_t n : scanSizes) {
    for (bool inPlace : {false, true}) {
      if (inPlace && n != scanSizes.back()) continue;
      t.check("provided scan" + std::string(inPlace ? " (in place)" : "") + ", n = " + fmtN(n), [&]() -> std::string {
        std::vector<uint32_t> v(n), ref(n);
        uint32_t x = n * 2654435761u + 7u, run = 0;
        for (uint32_t i = 0; i < n; ++i) {
          x ^= x << 13; x ^= x >> 17; x ^= x << 5;
          v[i] = x & 3u;
          ref[i] = run;
          run += v[i];
        }
        Buf* in = pmpp::alloc((size_t)n * 4);
        Buf* out = inPlace ? in : pmpp::alloc((size_t)n * 4);
        Buf* scratch = pmpp::alloc((size_t)scanScratchElems(n) * 4);
        pmpp::upload(in, v.data(), (size_t)n * 4);
        std::string err = runScan(in, out, n, scratch);
        std::vector<uint32_t> got(n);
        pmpp::download(out, got.data(), (size_t)n * 4);
        pmpp::release(scratch);
        if (!inPlace) pmpp::release(out);
        pmpp::release(in);
        if (!err.empty()) return err;
        for (uint32_t i = 0; i < n; ++i) {
          if (got[i] != ref[i]) {
            char b[160];
            std::snprintf(b, sizeof b, "element %u is %u, expected %u (the provided scan is broken)", i, got[i], ref[i]);
            return b;
          }
        }
        return "";
      });
    }
  }

  // ---- your filters
  const std::vector<uint32_t> devSizes = {1, 31, 32, 33, 255, 256, 257, 1000, 4099, 65537, 1u << 20, 3u * (1u << 20) + 17};
  const std::vector<uint32_t> raceSizes = {1, 33, 257, 1000, 2048};
  std::vector<int> parts = {0};
  for (int c : kCoarse) parts.push_back(c);
  for (int coarse : parts) {
    if (coarse > 0) {
      std::string why = stagedProblem(coarse);
      if (!why.empty() && why.find("did not build") == std::string::npos) {
        std::printf("  skipped  %s: %s\n", partName(coarse).c_str(), why.c_str());
        continue;
      }
    }
    std::vector<uint32_t> sizes = race ? raceSizes : devSizes;
    if (emu) {   // the emulator runs every thread on the CPU: smaller sizes, 65537 for part A only
      sizes = coarse == 0 ? std::vector<uint32_t>{1, 31, 32, 33, 255, 256, 257, 1000, 4099, 65537}
                          : std::vector<uint32_t>{1, 31, 33, 257, 1000, 4099};
    }
    for (uint32_t n : sizes) {
      // On the emulator, n = 65537 runs at half kept only (each fraction takes seconds there).
      const std::vector<double> fractions = emu && n > 4099 ? std::vector<double>{0.5} : kFractions;
      t.check(partName(coarse) + ", n = " + fmtN(n) + (fractions.size() > 1 ? ", kept fractions 0, 0.001, 0.5, 0.97, 1"
                                                                            : ", kept fraction 0.5"),
              [&]() -> std::string {
        Problem p(n);
        for (size_t k = 0; k < fractions.size(); ++k) {
          const std::vector<uint32_t> keys = makeKeys(n, KeepRule{Random, fractions[k], 1, 0}, 31 * n + (uint32_t)k);
          p.prepare(keys);
          std::string err = runPart(p, coarse);
          if (err.empty()) err = p.check(keys);
          if (!err.empty()) {
            char f[32];
            std::snprintf(f, sizeof f, "%g", fractions[k]);
            return "kept fraction " + std::string(f) + ": " + err;
          }
        }
        return "";
      });
    }
    // Patterns, on a few sizes. A block of part A owns 256 keys, one of part B 256 * coarse.
    const uint32_t tile = kBlock * (uint32_t)(coarse == 0 ? 1 : coarse);
    std::vector<uint32_t> psizes = race ? std::vector<uint32_t>{257, 1000, 2048}
                                 : emu ? std::vector<uint32_t>{257, 4099}
                                       : std::vector<uint32_t>{257, 4099, 65537, 3u * (1u << 20) + 17};
    for (Pattern pat : {FirstOnly, LastOnly, EveryOther, RunsAcrossTiles, OneFullTile}) {
      // The two tile patterns need at least one whole tile (and a boundary to cross).
      std::vector<uint32_t> use;
      for (uint32_t n : psizes)
        if ((pat != RunsAcrossTiles || n >= tile + 7) && (pat != OneFullTile || n >= tile)) use.push_back(n);
      if (use.empty()) continue;   // e.g. race at coarse 16: every n there is below one tile
      std::string sizeList;
      for (uint32_t n : use) sizeList += (sizeList.empty() ? "" : ", ") + fmtN(n);
      t.check(partName(coarse) + ", " + patternName(pat) + ", n = " + sizeList,
              [&, use]() -> std::string {
                for (uint32_t n : use) {
                  KeepRule r{pat, 0.0, tile, n >= 2 * tile ? 1u : 0u};
                  const std::vector<uint32_t> keys = makeKeys(n, r, 77 + n);
                  Problem p(n);
                  p.prepare(keys);
                  std::string err = runPart(p, coarse);
                  if (err.empty()) err = p.check(keys);
                  if (!err.empty()) return "n = " + fmtN(n) + ": " + err;
                }
                return "";
              });
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// The lesson's byte counts per key (section 12.5), for kept fraction f:
//   part A, named arrays: input read twice (8), keep written once and read twice (12), offset
//   written once and read once (8), kept keys written (4f): 28 + 4f, 30 at half kept.
//   part B: input read twice (8), kept keys written (4f); the per-block counts are negligible:
//   8 + 4f, about 10 at half kept.
// The provided scan really moves 16 bytes per scanned element, not the 8 the model charges for
// it (scan-scan-add, p. 274), so part A's actual traffic is about 8 bytes per key above its model.
static double modelBytesPerKey(int coarse, double f) { return coarse == 0 ? 28.0 + 4.0 * f : 8.0 + 4.0 * f; }

static int bench(const pmpp::Args& a) {
  // Size: start at 2^24 keys and double until a plain copy of the input takes a few
  // milliseconds, within an eighth of the device's memory (part A needs about 16 bytes per key).
  const double mem = deviceMemory();
  uint32_t n = 1u << 24;
  double copyS = 0;
  for (;;) {
    Buf *in = pmpp::alloc((size_t)n * 4), *out = pmpp::alloc((size_t)n * 4);
    copyS = pmpp::time_it(a.reps, [&] { return runCopy(in, out, n); }).min_s;
    pmpp::release(in); pmpp::release(out);
    const bool roomToGrow = mem <= 0 || 2.0 * n * 16.0 < mem / 8;
    if (copyS >= 3e-3 || !roomToGrow || n >= (1u << 28)) break;
    n *= 2;
  }
  std::string err = pmpp::take_error();
  if (!err.empty() || copyS <= 0) { std::printf("the copy reference failed to run: %s\n", err.c_str()); return 1; }
  const double copyGBps = 8.0 * n / copyS / 1e9;
  std::printf("n = %u keys (%.0f MB), best of %d runs. Copy reference: %.3f ms, %.1f GB/s (8 bytes per key).\n",
              n, n * 4.0 / 1e6, a.reps, copyS * 1e3, copyGBps);
  std::printf("GB/s counts the minimum traffic, 4n + 4 x kept + 4 bytes. \"Model\" is the lesson's bytes per key;\n"
              "\"at copy speed\" is how many bytes the copy would have moved in the same time, per key.\n\n");
  std::printf("  %-8s %-6s %-6s %10s %8s %10s %14s\n", "fraction", "kernel", "coarse", "ms", "GB/s", "model B/key",
              "at copy speed");
  pmpp::bench_row(a, "copy").n("n", n).n("seconds", copyS).n("gbps", copyGBps).emit();

  Problem p(n);
  int rc = 0;
  for (double f : {0.001, 0.01, 0.1, 0.5, 0.9, 1.0}) {
    const std::vector<uint32_t> keys = makeKeys(n, KeepRule{Random, f, 1, 0}, 2024);
    size_t kept = 0;
    for (uint32_t k : keys) kept += k < kThreshold;
    const double keptF = (double)kept / n;
    const double minBytes = 4.0 * n + 4.0 * kept + 4.0;
    std::vector<int> parts = {0};
    for (int c : kCoarse) parts.push_back(c);
    for (int coarse : parts) {
      if (coarse > 0 && !stagedProblem(coarse).empty()) continue;   // does not fit this device
      p.prepare(keys);
      std::string why = runPart(p, coarse);
      if (why.empty()) why = p.check(keys);   // every timed result is checked first
      if (!why.empty()) {
        std::printf("  %-8g %s: not timed, the result is wrong: %s\n", f, partName(coarse).c_str(), why.c_str());
        rc = 1;
        continue;
      }
      const double s = pmpp::time_it(a.reps, [&] {
                         pmpp::Timing tm{0, 0};
                         if (coarse == 0) runFlags(p.in, p.out, n, p.size, p.keep, p.offset, p.scratch, &tm);
                         else runStaged(p.in, p.out, n, p.size, p.counts, p.offsets, p.scratch, coarse, &tm);
                         return tm;
                       }).min_s;
      err = pmpp::take_error();
      if (!err.empty() || s <= 0) { std::printf("  %s failed: %s\n", partName(coarse).c_str(), err.c_str()); rc = 1; continue; }
      const double gbps = minBytes / s / 1e9, model = modelBytesPerKey(coarse, keptF), atCopy = 8.0 * s / copyS;
      std::printf("  %-8g %-6s %-6s %10.4f %8.1f %10.1f %14.1f\n", f, coarse == 0 ? "flags" : "staged",
                  coarse == 0 ? "-" : std::to_string(coarse).c_str(), s * 1e3, gbps, model, atCopy);
      pmpp::bench_row(a, coarse == 0 ? "flags" : "staged")
          .n("fraction", f).n("kept_fraction", keptF).n("coarse", coarse).n("n", n).n("seconds", s).n("gbps", gbps)
          .n("model_bytes_per_key", model).n("copy_equivalent_bytes_per_key", atCopy).emit();
    }
  }
  std::printf("\nThese are measurements of this device at this moment, not targets.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
