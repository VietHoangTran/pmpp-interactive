// Lab 14.1 driver: tests your odd-even sort (part A) and your one-bit radix sort of
// (key, value) pairs (part B) against the C++ standard library, tests the provided scan on its
// own, and benchmarks both sorts and one CPU thread's std::stable_sort. You do not need to edit
// this file; your work goes in work/.
#include <chrono>
#include <cstring>
#include <map>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static const uint32_t kGarbage = 0xCDCDCDCDu;   // fills scratch, bits and every guard slot before a run
static const uint32_t kPad = 64;                // guard slots past the end of every array

// ---- entry points implemented in <variant>/ and ../shared-g11/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
int runOddEvenSort(id<MTLCommandQueue> queue, id<MTLComputePipelineState> step, id<MTLBuffer> data,
                   id<MTLBuffer> flag, uint32_t n);
bool encodeRadixSortOneBit(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> extract,
                           id<MTLComputePipelineState> scatter, id<MTLBuffer> keys, id<MTLBuffer> vals,
                           id<MTLBuffer> tmpKeys, id<MTLBuffer> tmpVals, id<MTLBuffer> bits,
                           id<MTLBuffer> scanScratch, uint32_t n);
int scanScratchCount(int count);
bool encodeExclusiveScan(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> data, uint32_t count, id<MTLBuffer> scratch);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso(const char* name) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, name); }
static id<MTLComputeCommandEncoder> encoderOf(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }
// The odd-even loop commits its own command buffers, one per pair of steps, on this queue.
static id<MTLCommandQueue> labQueue() {
  static id<MTLCommandQueue> q = [pmpp::metal::device() newCommandQueue];
  return q;
}

// Runs your odd-even sort. *steps: what it returned; *seconds: wall time of the whole loop.
static std::string runOddEven(Buf* data, Buf* flag, uint32_t n, int* steps, double* seconds) {
  id<MTLComputePipelineState> p = pso("odd_even_step");
  if (!p) return "odd_even_step did not build: " + pmpp::take_error();
  const auto t0 = std::chrono::steady_clock::now();
  *steps = runOddEvenSort(labQueue(), p, buffer(data), buffer(flag), n);
  *seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  if (*steps < 0) return "runOddEvenSort returned -1: a command buffer failed";
  return "";
}
static std::string runRadix(Buf* keys, Buf* vals, Buf* tmpKeys, Buf* tmpVals, Buf* bits, Buf* scratch, uint32_t n,
                            pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> extract = pso("radix_extract_bits"), scatter = pso("radix_scatter");
  if (!extract || !scatter) return "a part B Metal kernel did not build: " + pmpp::take_error();
  if (extract.maxTotalThreadsPerThreadgroup < 256 || scatter.maxTotalThreadsPerThreadgroup < 256)
    return "these pipelines cannot run 256 threads per threadgroup";
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    ok = encodeRadixSortOneBit(encoderOf(v), extract, scatter, buffer(keys), buffer(vals), buffer(tmpKeys),
                               buffer(tmpVals), buffer(bits), buffer(scratch), n);
  });
  if (tm) *tm = t;
  return ok ? "" : "encodeRadixSortOneBit returned false, although its pipelines can run 256 threads per threadgroup";
}
static std::string runScan(Buf* data, uint32_t count, Buf* scratch) {
  bool ok = true;
  pmpp::run([&](void* v) { ok = encodeExclusiveScan(encoderOf(v), buffer(data), count, buffer(scratch)); });
  return ok ? "" : "the provided scan did not run: " + pmpp::take_error();
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
int launchOddEvenSort(unsigned int* data, unsigned int* flag, int n);
void launchRadixSortOneBit(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys, unsigned int* tmpVals,
                           unsigned int* bits, unsigned int* scanScratch, int n);
int scanScratchCount(int count);
void launchExclusiveScan(unsigned int* data, int count, unsigned int* scratch);
static unsigned int* u(Buf* b) { return (unsigned int*)pmpp::raw(b); }

static std::string runOddEven(Buf* data, Buf* flag, uint32_t n, int* steps, double* seconds) {
  pmpp::Timing t = pmpp::run([&](void*) { *steps = launchOddEvenSort(u(data), u(flag), (int)n); });
  *seconds = t.wall_s;
  return "";
}
static std::string runRadix(Buf* keys, Buf* vals, Buf* tmpKeys, Buf* tmpVals, Buf* bits, Buf* scratch, uint32_t n,
                            pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchRadixSortOneBit(u(keys), u(vals), u(tmpKeys), u(tmpVals), u(bits), u(scratch), (int)n);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runScan(Buf* data, uint32_t count, Buf* scratch) {
  pmpp::run([&](void*) { launchExclusiveScan(u(data), (int)count, u(scratch)); });
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- helpers
static std::string fmtN(uint32_t n) {
  for (int p = 16; p <= 26; ++p)
    if (n == (1u << p)) return "2^" + std::to_string(p);
  if (n == (1u << 20) + 3) return "2^20 + 3";
  return std::to_string(n);
}
static uint32_t xorshift(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
static void fillBuf(Buf* b, size_t elems, uint32_t v) {
  std::vector<uint32_t> h(elems, v);
  pmpp::upload(b, h.data(), elems * 4);
}
static std::vector<uint32_t> downloadU32(Buf* b, size_t elems) {
  std::vector<uint32_t> h(elems);
  pmpp::download(b, h.data(), elems * 4);
  return h;
}
static std::string fmt(const char* f, uint32_t a, uint32_t b = 0, uint32_t c = 0, uint32_t d = 0) {
  char buf[400];
  std::snprintf(buf, sizeof buf, f, a, b, c, d);
  return buf;
}

// ---------------------------------------------------------------- part A: odd-even sort
// Inputs. "Largest first" is the book's worst case (p. 333): the largest key must travel the
// whole list. "Trap" is sorted except for one pair at an odd index, so the first (even) step
// makes no swap although the list is not sorted.
enum OddEvenInput { OeRandom, OeReversed, OeSorted, OeEqual, OeLargestFirst, OeTrap };
static const std::vector<OddEvenInput> kOddEvenInputs = {OeRandom, OeReversed, OeSorted, OeEqual, OeLargestFirst, OeTrap};
static const char* oddEvenName(OddEvenInput k) {
  switch (k) {
    case OeReversed: return "reversed";
    case OeSorted: return "already sorted";
    case OeEqual: return "all equal";
    case OeLargestFirst: return "largest first";
    case OeTrap: return "trap [1, 3, 2, 4, ...]";
    default: return "random";
  }
}
static std::vector<uint32_t> oddEvenData(OddEvenInput k, uint32_t n, uint32_t seed) {
  std::vector<uint32_t> v(n);
  uint32_t x = seed * 2654435761u + 99u;
  for (uint32_t i = 0; i < n; ++i) {
    switch (k) {
      case OeReversed: v[i] = (n - i) * 7919u; break;
      case OeSorted: v[i] = i * 7919u + 3u; break;
      case OeEqual: v[i] = 0x9E3779B9u; break;
      case OeLargestFirst: v[i] = i == 0 ? 0xFFFFFFFFu : i; break;
      case OeTrap: v[i] = i + 1; break;
      default: v[i] = xorshift(x); break;
    }
  }
  if (k == OeTrap && n >= 3) std::swap(v[1], v[2]);
  return v;
}
// A correct loop stops after the first quiet even-odd pair of steps. Odd-even transposition
// sort needs at most n steps; the pair the last swap is in, plus one quiet pair, give at most
// n + 2 steps when n is even and n + 3 when n is odd (every pair counts 2).
static int oddEvenStepBound(uint32_t n) { return n < 2 ? 2 : (int)(2 * ((n + 1) / 2) + 2); }

// Sorts one input and returns "" if the result is right.
static std::string oddEvenCase(OddEvenInput k, uint32_t n, uint32_t seed, int* stepsOut = nullptr,
                               double* secondsOut = nullptr) {
  const std::vector<uint32_t> in = oddEvenData(k, n, seed);
  // Guard slots past the end hold 0, smaller than every key: a pair that reaches past the end
  // swaps a key into them, and the check below sees it.
  std::vector<uint32_t> host(in);
  host.resize(n + kPad, 0u);
  Buf* data = pmpp::alloc((size_t)(n + kPad) * 4);
  Buf* flag = pmpp::alloc(16);
  pmpp::upload(data, host.data(), host.size() * 4);
  fillBuf(flag, 4, kGarbage);
  int steps = -1;
  double seconds = 0;
  std::string err = runOddEven(data, flag, n, &steps, &seconds);
  const std::vector<uint32_t> got = downloadU32(data, n + kPad);
  pmpp::release(data);
  pmpp::release(flag);
  if (stepsOut) *stepsOut = steps;
  if (secondsOut) *secondsOut = seconds;
  if (!err.empty()) return err;
  for (uint32_t i = n; i < n + kPad; ++i)
    if (got[i] != 0u) return fmt("data[%u] past the end (n = %u) was written: a pair reached past the last element", i, n);
  std::vector<uint32_t> ref(in);
  std::sort(ref.begin(), ref.end());
  for (uint32_t i = 0; i + 1 < n; ++i) {
    if (got[i] > got[i + 1]) {
      return fmt("not sorted: data[%u] = %u > data[%u] = %u", i, got[i], i + 1, got[i + 1]) +
             " (" + std::to_string(steps) + " steps launched)";
    }
  }
  for (uint32_t i = 0; i < n; ++i)
    if (got[i] != ref[i]) return fmt("sorted, but not a permutation of the input: data[%u] = %u, expected %u", i, got[i], ref[i]);
  if (steps < 0 || steps > oddEvenStepBound(n) || steps % 2 != 0 || (n >= 2 && steps < 2)) {
    return "sorted, but the loop reported " + std::to_string(steps) +
           " steps; a loop that stops at the first quiet even-odd pair launches an even number, at least 2 and at most " +
           std::to_string(oddEvenStepBound(n));
  }
  return "";
}

// ---------------------------------------------------------------- part B: radix sort
enum KeyPattern { KRandom, KSmall, KEqual, KSorted, KReversed, KPowers };
static const std::vector<KeyPattern> kKeyPatterns = {KRandom, KSmall, KEqual, KSorted, KReversed, KPowers};
static const char* keyName(KeyPattern p) {
  switch (p) {
    case KSmall: return "keys 0..15";
    case KEqual: return "all equal";
    case KSorted: return "sorted";
    case KReversed: return "reversed";
    case KPowers: return "powers of two";
    default: return "random, full range";
  }
}
// Full-range keys have the top bit set about half the time, so a signed key or a shift that
// stops short of bit 31 shows up.
static std::vector<uint32_t> radixKeys(KeyPattern p, uint32_t n, uint32_t seed) {
  std::vector<uint32_t> k(n);
  uint32_t x = seed * 2654435761u + 12345u;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t r = xorshift(x);
    switch (p) {
      case KSmall: k[i] = r & 15u; break;
      case KEqual: k[i] = 0xA5A5A5A5u; break;
      case KPowers: k[i] = 1u << (r % 32u); break;
      default: k[i] = r; break;
    }
  }
  if (p == KSorted) std::sort(k.begin(), k.end());
  if (p == KReversed) std::sort(k.begin(), k.end(), [](uint32_t a, uint32_t b) { return a > b; });
  return k;
}

// One CPU reference: std::stable_sort of (key, input position) by key.
static void cpuStableSort(const std::vector<uint32_t>& keys, std::vector<uint32_t>& outKeys, std::vector<uint32_t>& outVals) {
  std::vector<std::pair<uint32_t, uint32_t>> pairs(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) pairs[i] = {keys[i], (uint32_t)i};
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) { return a.first < b.first; });
  outKeys.resize(keys.size());
  outVals.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) { outKeys[i] = pairs[i].first; outVals[i] = pairs[i].second; }
}

// Device buffers for one radix sort of n pairs, each with kPad guard slots past the end.
struct RadixProblem {
  uint32_t n;
  Buf *keys, *vals, *tmpKeys, *tmpVals, *bits, *scratch;
  int scratchCount;
  explicit RadixProblem(uint32_t n_) : n(n_) {
    scratchCount = scanScratchCount((int)n + 1);
    keys = pmpp::alloc((size_t)(n + kPad) * 4);
    vals = pmpp::alloc((size_t)(n + kPad) * 4);
    tmpKeys = pmpp::alloc((size_t)(n + kPad) * 4);
    tmpVals = pmpp::alloc((size_t)(n + kPad) * 4);
    bits = pmpp::alloc((size_t)(n + 1 + kPad) * 4);
    scratch = pmpp::alloc((size_t)scratchCount * 4);
  }
  ~RadixProblem() { for (Buf* b : {keys, vals, tmpKeys, tmpVals, bits, scratch}) pmpp::release(b); }
  RadixProblem(const RadixProblem&) = delete;
  // Input pairs (key, input position); garbage in every other slot, bits[n] and the scratch included.
  void prepare(const std::vector<uint32_t>& k) {
    std::vector<uint32_t> h(n + kPad, kGarbage);
    std::copy(k.begin(), k.end(), h.begin());
    pmpp::upload(keys, h.data(), h.size() * 4);
    for (uint32_t i = 0; i < n; ++i) h[i] = i;
    pmpp::upload(vals, h.data(), h.size() * 4);
    fillBuf(tmpKeys, n + kPad, kGarbage);
    fillBuf(tmpVals, n + kPad, kGarbage);
    fillBuf(bits, n + 1 + kPad, kGarbage);
    fillBuf(scratch, (size_t)scratchCount, kGarbage);
  }
  std::string run(pmpp::Timing* tm = nullptr) { return runRadix(keys, vals, tmpKeys, tmpVals, bits, scratch, n, tm); }
  // "" when keys/vals equal std::stable_sort's keys and permutation.
  std::string check(const std::vector<uint32_t>& refKeys, const std::vector<uint32_t>& refVals) {
    const std::vector<uint32_t> gk = downloadU32(keys, n + kPad), gv = downloadU32(vals, n + kPad);
    const std::vector<uint32_t> tk = downloadU32(tmpKeys, n + kPad), tv = downloadU32(tmpVals, n + kPad);
    const std::vector<uint32_t> gb = downloadU32(bits, n + 1 + kPad);
    const char* names[4] = {"keys", "vals", "tmpKeys", "tmpVals"};
    const std::vector<uint32_t>* arrays[4] = {&gk, &gv, &tk, &tv};
    for (int a = 0; a < 4; ++a)
      for (uint32_t i = n; i < n + kPad; ++i)
        if ((*arrays[a])[i] != kGarbage)
          return std::string(names[a]) + fmt("[%u] was written, but only [0, %u) belongs to the sort", i, n);
    for (uint32_t i = n + 1; i < n + 1 + kPad; ++i)
      if (gb[i] != kGarbage) return fmt("bits[%u] was written, but bits has n + 1 = %u entries", i, n + 1);
    for (uint32_t i = 0; i < n; ++i) {
      if (gk[i] != refKeys[i]) {
        if (gk[i] == kGarbage) return fmt("keys[%u] is 0xCDCDCDCD, the garbage the driver wrote: no pass wrote it", i);
        return fmt("keys[%u] = 0x%08x, std::stable_sort has 0x%08x there", i, gk[i], refKeys[i]);
      }
    }
    for (uint32_t i = 0; i < n; ++i) {
      if (gv[i] != refVals[i]) {
        if (gv[i] == kGarbage) return fmt("the keys are right, but vals[%u] was never written", i);
        return fmt("the keys are right, but vals[%u] holds input position %u where std::stable_sort has %u: "
                   "equal keys left their input order, or a value did not move with its key", i, gv[i], refVals[i]);
      }
    }
    return "";
  }
};

static std::string radixCase(KeyPattern p, uint32_t n, uint32_t seed) {
  const std::vector<uint32_t> k = radixKeys(p, n, seed);
  std::vector<uint32_t> refKeys, refVals;
  cpuStableSort(k, refKeys, refVals);
  RadixProblem prob(n);
  prob.prepare(k);
  std::string err = prob.run();
  if (err.empty()) err = prob.check(refKeys, refVals);
  return err;
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  const bool race = be == "race", emu = be == "emu";

  // ---- the provided scan on its own (it is not yours: a failure here is a broken provided file)
  const std::vector<uint32_t> scanSizes = race ? std::vector<uint32_t>{1, 2, 1025, 2049}
                                        : emu ? std::vector<uint32_t>{1, 2, 1023, 1024, 1025, 4097, 65537}
                                              : std::vector<uint32_t>{1, 2, 1023, 1024, 1025, 4097, 65537, (1u << 20) + 3};
  for (uint32_t count : scanSizes) {
    t.check("provided scan, " + fmtN(count) + " elements", [&]() -> std::string {
      std::vector<uint32_t> v(count), ref(count);
      uint32_t x = count * 2654435761u + 7u, run = 0;
      for (uint32_t i = 0; i < count; ++i) {
        v[i] = xorshift(x) & 3u;
        ref[i] = run;
        run += v[i];
      }
      Buf* data = pmpp::alloc((size_t)count * 4);
      const int sc = scanScratchCount((int)count);
      Buf* scratch = pmpp::alloc((size_t)sc * 4);
      pmpp::upload(data, v.data(), (size_t)count * 4);
      fillBuf(scratch, (size_t)sc, kGarbage);
      std::string err = runScan(data, count, scratch);
      const std::vector<uint32_t> got = downloadU32(data, count);
      pmpp::release(scratch);
      pmpp::release(data);
      if (!err.empty()) return err;
      for (uint32_t i = 0; i < count; ++i)
        if (got[i] != ref[i]) return fmt("element %u is %u, expected %u (the provided scan is broken)", i, got[i], ref[i]);
      return "";
    });
  }

  // ---- part A: odd-even sort. n = 1 is checked together with n = 2: one element is sorted
  // whatever the loop does, so on its own it would prove nothing.
  const std::vector<std::vector<uint32_t>> oeSizes =
      race ? std::vector<std::vector<uint32_t>>{{1, 2}, {3}, {4}, {8}, {33}, {64}}
           : emu ? std::vector<std::vector<uint32_t>>{{1, 2}, {3}, {4}, {8}, {33}, {256}}
                 : std::vector<std::vector<uint32_t>>{{1, 2}, {3}, {4}, {8}, {33}, {256}, {1000}};
  for (const auto& group : oeSizes) {
    std::string label = "part A, odd-even sort, n = ";
    for (size_t g = 0; g < group.size(); ++g) label += (g ? " and " : "") + std::to_string(group[g]);
    t.check(label + ": random, reversed, sorted, all equal, largest first, trap", [&]() -> std::string {
      for (uint32_t n : group) {
        for (OddEvenInput k : kOddEvenInputs) {
          std::string err = oddEvenCase(k, n, 17 + n);
          if (!err.empty()) return "n = " + std::to_string(n) + ", " + oddEvenName(k) + ": " + err;
        }
      }
      return "";
    });
  }

  // ---- part B: one-bit radix sort of (key, input position) pairs. Every size runs every key
  // pattern, except on the emulator: it pays for every thread of every block of 96 launches per
  // sort, so there every pattern runs at n = 1000 and two patterns at each other size.
  struct RadixGroup { std::vector<uint32_t> sizes; std::vector<KeyPattern> patterns; };
  std::vector<RadixGroup> rxGroups;
  if (emu) {
    rxGroups = {{{1, 2}, {KReversed, KRandom}}, {{31}, {KRandom, KSmall}}, {{32}, {KPowers, KSorted}},
                {{33}, {KEqual, KReversed}},    {{1000}, kKeyPatterns},   {{4097}, {KRandom, KSmall}}};
  } else {
    std::vector<std::vector<uint32_t>> sizes = {{1, 2}, {31}, {32}, {33}, {1000}};
    if (!race) for (uint32_t n : {4097u, 1u << 16, 1u << 20}) sizes.push_back({n});
    for (auto& s : sizes) rxGroups.push_back({s, kKeyPatterns});
  }
  for (const RadixGroup& group : rxGroups) {
    std::string label = "part B, one-bit radix sort, n = ";
    for (size_t g = 0; g < group.sizes.size(); ++g) label += (g ? " and " : "") + fmtN(group.sizes[g]);
    std::string pats;
    for (KeyPattern p : group.patterns) pats += (pats.empty() ? "" : ", ") + std::string(keyName(p));
    t.check(label + ": " + (group.patterns.size() == kKeyPatterns.size() ? std::string("6 key patterns") : pats),
            [&]() -> std::string {
      for (uint32_t n : group.sizes) {
        for (KeyPattern p : group.patterns) {
          std::string err = radixCase(p, n, 1000 + n + (uint32_t)p);
          if (!err.empty()) return "n = " + fmtN(n) + ", " + keyName(p) + ": " + err;
        }
      }
      return "";
    });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
static double cpuStableSortSeconds(const std::vector<uint32_t>& keys, int reps) {
  std::vector<uint32_t> rk, rv;
  double best = 1e30;
  for (int r = 0; r <= reps; ++r) {   // the first run is a warm-up
    const auto t0 = std::chrono::steady_clock::now();
    cpuStableSort(keys, rk, rv);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (r > 0) best = std::min(best, s);
  }
  return best;
}

static int bench(const pmpp::Args& a) {
  int rc = 0;
  const double mem = deviceMemory();
  std::printf("Best of %d runs after one warm-up. Every timed result is checked first.\n", a.reps);
  std::printf("Odd-even: wall time of the whole host loop (launches, flag read-backs).\n");
  std::printf("Radix: device time of the command buffer on Metal; wall time around the launches and the\n"
              "final synchronisation on CUDA and HIP. \"wall\" is always the host's clock around the sort.\n");
  std::printf("CPU: std::stable_sort of the same (key, position) pairs on one CPU thread (a host measurement).\n\n");

  // ---- odd-even sort
  std::map<uint32_t, double> oddEvenWall;
  for (uint32_t n : {1024u, 4096u}) {
    int steps = 0;
    double best = 1e30;
    std::string err;
    for (int r = 0; r <= a.reps && err.empty(); ++r) {
      double s = 0;
      err = oddEvenCase(OeRandom, n, 5, &steps, &s);   // sorts, then checks the result
      if (r > 0) best = std::min(best, s);
    }
    if (!err.empty()) {
      std::printf("  odd-even, n = %u: not timed, the result is wrong: %s\n", n, err.c_str());
      rc = 1;
      continue;
    }
    oddEvenWall[n] = best;
    std::printf("  odd-even     n = %-8s %6d steps  %10.3f ms\n", fmtN(n).c_str(), steps, best * 1e3);
    pmpp::bench_row(a, "oddeven").n("n", n).n("steps", steps).n("seconds", best).emit();
  }
  std::printf("\n");

  // ---- one-bit radix sort and the CPU
  std::printf("  %-12s %-8s %12s %12s %12s %16s %16s\n", "", "n", "ms", "wall ms", "Mkeys/s", "odd-even/radix", "radix vs CPU");
  for (uint32_t n : {1024u, 4096u, 1u << 16, 1u << 20, 1u << 22}) {
    // Six arrays of n + 1 pairs, the scan's scratch and the host copies: stay well inside the device.
    if (mem > 0 && 30.0 * n > mem / 8) {
      std::printf("  radix, n = %s: skipped, it would use more than an eighth of the device's memory\n", fmtN(n).c_str());
      continue;
    }
    const std::vector<uint32_t> k = radixKeys(KRandom, n, 2024);
    std::vector<uint32_t> refKeys, refVals;
    cpuStableSort(k, refKeys, refVals);
    RadixProblem prob(n);
    double best = 1e30, bestWall = 1e30;
    std::string err;
    for (int r = 0; r <= a.reps && err.empty(); ++r) {
      prob.prepare(k);
      pmpp::Timing tm{0, -1};
      err = prob.run(&tm);
      if (err.empty()) err = pmpp::take_error();
      if (err.empty()) err = prob.check(refKeys, refVals);
      if (r > 0) { best = std::min(best, tm.best()); bestWall = std::min(bestWall, tm.wall_s); }
    }
    if (!err.empty() || best <= 0) {
      std::printf("  radix, n = %s: not timed, the result is wrong: %s\n", fmtN(n).c_str(), err.c_str());
      rc = 1;
      continue;
    }
    const double cpu = cpuStableSortSeconds(k, a.reps);
    char ratio[32] = "";
    if (oddEvenWall.count(n)) std::snprintf(ratio, sizeof ratio, "%.1fx slower", oddEvenWall[n] / bestWall);
    char vsCpu[32];
    std::snprintf(vsCpu, sizeof vsCpu, "%.2fx", cpu / best);
    std::printf("  %-12s %-8s %12.3f %12.3f %12.1f %16s %16s\n", "radix 1-bit", fmtN(n).c_str(), best * 1e3,
                bestWall * 1e3, n / best / 1e6, ratio, vsCpu);
    std::printf("  %-12s %-8s %12.3f %12s %12.1f\n", "CPU stable", fmtN(n).c_str(), cpu * 1e3, "", n / cpu / 1e6);
    pmpp::bench_row(a, "radix1").n("n", n).n("seconds", best).n("wall_seconds", bestWall)
        .n("mkeys_per_s", n / best / 1e6).emit();
    pmpp::bench_row(a, "cpu_stable_sort").n("n", n).n("seconds", cpu).n("mkeys_per_s", n / cpu / 1e6).emit();
  }
  std::printf("\n\"odd-even/radix\" divides the two wall times at the same n. These are measurements of this\n"
              "machine at this moment, not targets.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
