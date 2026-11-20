// Lab 14.2 driver: tests your multi-bit radix sort with local sorting (exercises 1 to 3 of
// chapter 14) against std::stable_sort for every radix width r and keys-per-thread c the lab
// names, tests the provided one-bit baseline, and benchmarks both and one CPU thread's
// std::stable_sort. You do not need to edit this file; your work goes in work/.
#include <chrono>
#include <cstring>
#include <map>
#include <memory>

#include "pmpp/harness.hpp"
using pmpp::Buf;

static const uint32_t kGarbage = 0xCDCDCDCDu;   // fills scratch, the table, localStarts and guard slots
static const uint32_t kPad = 64;                // guard slots past the end of every pair array
static const uint32_t kBlock = 256;             // BLOCK_DIM in kernel.cu and kernel.metal
static const std::vector<int> kTestRadix = {1, 2, 3, 4, 5, 8};

// ---- entry points implemented in <variant>/ and ../shared-g11/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeRadixSortMulti(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> localSort,
                          id<MTLComputePipelineState> scatterLocal, id<MTLBuffer> keys, id<MTLBuffer> vals,
                          id<MTLBuffer> tmpKeys, id<MTLBuffer> tmpVals, id<MTLBuffer> table,
                          id<MTLBuffer> localStarts, id<MTLBuffer> scanScratch, uint32_t n, uint32_t radixBits,
                          uint32_t keysPerThread);
bool encodeRadixSortOneBitGlobal(id<MTLComputeCommandEncoder> enc, id<MTLBuffer> keys, id<MTLBuffer> vals,
                                 id<MTLBuffer> tmpKeys, id<MTLBuffer> tmpVals, id<MTLBuffer> bits,
                                 id<MTLBuffer> scanScratch, uint32_t n);
int scanScratchCount(int count);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso(const char* name) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, name); }
static id<MTLComputeCommandEncoder> encoderOf(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }

// Why (r, c) cannot run on this device, or "". The same rule a correct launch.mm applies:
// 256 threads in whole SIMD-groups, and the tile (256c pairs) plus one slot per SIMD-group plus
// the kernel's static arrays within the device's threadgroup memory (ch05-facts C1, C3, C8).
static std::string multiProblem(int r, int c) {
  (void)r;
  id<MTLComputePipelineState> ls = pso("radix_local_sort"), sc = pso("radix_scatter_local");
  if (!ls || !sc) return "a Metal kernel did not build: " + pmpp::take_error();
  const NSUInteger w = ls.threadExecutionWidth;
  if (kBlock > ls.maxTotalThreadsPerThreadgroup || kBlock > sc.maxTotalThreadsPerThreadgroup || w == 0 || kBlock % w)
    return "these pipelines cannot run 256 threads per threadgroup in whole SIMD-groups";
  const double need = (double)ls.staticThreadgroupMemoryLength + ((2.0 * kBlock * c * 4 + 15) / 16) * 16 +
                      ((kBlock / w * 4 + 15) / 16) * 16;
  const double have = pmpp::device_prop("maxThreadgroupMemoryLength", 0);
  if (need > have) {
    char b[160];
    std::snprintf(b, sizeof b, "needs %.0f bytes of threadgroup memory, the device offers %.0f", need, have);
    return b;
  }
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchRadixSortMulti(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys, unsigned int* tmpVals,
                          unsigned int* table, unsigned int* localStarts, unsigned int* scanScratch, int n,
                          int radixBits, int keysPerThread);
void launchRadixSortOneBitGlobal(unsigned int* keys, unsigned int* vals, unsigned int* tmpKeys, unsigned int* tmpVals,
                                 unsigned int* bits, unsigned int* scanScratch, int n);
int scanScratchCount(int count);
static unsigned int* u(Buf* b) { return (unsigned int*)pmpp::raw(b); }

// The tile (256c pairs, dynamic) plus the reference kernel's static arrays (a 512-entry scan
// buffer and 257 bucket starts) against the device's shared memory per block.
static std::string multiProblem(int r, int c) {
  (void)r;
  const double need = 2.0 * kBlock * c * 4 + (2.0 * kBlock + 257) * 4;
  const double have = pmpp::device_prop("sharedMemPerBlock", 48 * 1024);
  if (need > have) {
    char b[160];
    std::snprintf(b, sizeof b, "needs %.0f bytes of shared memory per block, the device offers %.0f", need, have);
    return b;
  }
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- helpers
static std::string fmtN(uint32_t n) {
  for (int p = 16; p <= 26; ++p)
    if (n == (1u << p)) return "2^" + std::to_string(p);
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
static std::string fmt(const char* f, uint32_t a, uint32_t b = 0, uint32_t c = 0) {
  char buf[400];
  std::snprintf(buf, sizeof buf, f, a, b, c);
  return buf;
}
static int passesFor(int r) { return (32 + r - 1) / r; }

// ---------------------------------------------------------------- keys
enum KeyPattern { KRandom, KSmall, KEqual, KSorted, KReversed, KPowers, KSameLowByte, KBelow256 };
static const std::vector<KeyPattern> kKeyPatterns = {KRandom, KSmall,  KEqual,       KSorted,
                                                     KReversed, KPowers, KSameLowByte, KBelow256};
static const char* keyName(KeyPattern p) {
  switch (p) {
    case KSmall: return "keys 0..15";
    case KEqual: return "all equal";
    case KSorted: return "sorted";
    case KReversed: return "reversed";
    case KPowers: return "powers of two";
    case KSameLowByte: return "every key shares its low byte";
    case KBelow256: return "keys below 256";
    default: return "random, full range";
  }
}
// "Shares its low byte": every block puts all its keys into one bucket on the first passes
// (r <= 8), so most of the table is zeros. "Below 256": every digit above bit 7 is zero, so the
// later passes put every key into bucket 0.
static std::vector<uint32_t> makeKeys(KeyPattern p, uint32_t n, uint32_t seed) {
  std::vector<uint32_t> k(n);
  uint32_t x = seed * 2654435761u + 12345u;
  for (uint32_t i = 0; i < n; ++i) {
    const uint32_t r = xorshift(x);
    switch (p) {
      case KSmall: k[i] = r & 15u; break;
      case KEqual: k[i] = 0xA5A5A5A5u; break;
      case KPowers: k[i] = 1u << (r % 32u); break;
      case KSameLowByte: k[i] = (r & 0xFFFFFF00u) | 0x5Au; break;
      case KBelow256: k[i] = r & 0xFFu; break;
      default: k[i] = r; break;
    }
  }
  if (p == KSorted) std::sort(k.begin(), k.end());
  if (p == KReversed) std::sort(k.begin(), k.end(), [](uint32_t a, uint32_t b) { return a > b; });
  return k;
}

// std::stable_sort of (key, input position) by key.
static void cpuStableSort(const std::vector<uint32_t>& keys, std::vector<uint32_t>& outKeys, std::vector<uint32_t>& outVals) {
  std::vector<std::pair<uint32_t, uint32_t>> pairs(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) pairs[i] = {keys[i], (uint32_t)i};
  std::stable_sort(pairs.begin(), pairs.end(),
                   [](const std::pair<uint32_t, uint32_t>& a, const std::pair<uint32_t, uint32_t>& b) { return a.first < b.first; });
  outKeys.resize(keys.size());
  outVals.resize(keys.size());
  for (size_t i = 0; i < keys.size(); ++i) { outKeys[i] = pairs[i].first; outVals[i] = pairs[i].second; }
}

// One input and its reference, built once per (pattern, n) and reused for every (r, c).
struct Case {
  std::vector<uint32_t> keys, refKeys, refVals;
};
static const Case& getCase(KeyPattern p, uint32_t n) {
  static std::map<std::pair<int, uint32_t>, Case> cache;
  auto key = std::make_pair((int)p, n);
  auto it = cache.find(key);
  if (it != cache.end()) return it->second;
  Case& c = cache[key];
  c.keys = makeKeys(p, n, 1000 + n + (uint32_t)p);
  cpuStableSort(c.keys, c.refKeys, c.refVals);
  return c;
}

// ---------------------------------------------------------------- device buffers for one n
// Sized for every configuration: the table and localStarts for the largest, 2^8 buckets of the
// most blocks (c = 1); bits for the provided one-bit baseline.
struct Problem {
  uint32_t n, tableMax;
  int scratchCount;
  Buf *keys, *vals, *tmpKeys, *tmpVals, *table, *localStarts, *bits, *scratch;
  explicit Problem(uint32_t n_) : n(n_) {
    tableMax = 256u * ((n + kBlock - 1) / kBlock);
    scratchCount = std::max(scanScratchCount((int)tableMax), scanScratchCount((int)n + 1));
    keys = pmpp::alloc((size_t)(n + kPad) * 4);
    vals = pmpp::alloc((size_t)(n + kPad) * 4);
    tmpKeys = pmpp::alloc((size_t)(n + kPad) * 4);
    tmpVals = pmpp::alloc((size_t)(n + kPad) * 4);
    table = pmpp::alloc((size_t)tableMax * 4);
    localStarts = pmpp::alloc((size_t)tableMax * 4);
    bits = pmpp::alloc((size_t)(n + 1) * 4);
    scratch = pmpp::alloc((size_t)scratchCount * 4);
  }
  ~Problem() { for (Buf* b : {keys, vals, tmpKeys, tmpVals, table, localStarts, bits, scratch}) pmpp::release(b); }
  Problem(const Problem&) = delete;
  // Input pairs (key, input position); garbage everywhere else, the table and localStarts included.
  void prepare(const std::vector<uint32_t>& k) {
    std::vector<uint32_t> h(n + kPad, kGarbage);
    std::copy(k.begin(), k.end(), h.begin());
    pmpp::upload(keys, h.data(), h.size() * 4);
    for (uint32_t i = 0; i < n; ++i) h[i] = i;
    pmpp::upload(vals, h.data(), h.size() * 4);
    fillBuf(tmpKeys, n + kPad, kGarbage);
    fillBuf(tmpVals, n + kPad, kGarbage);
    fillBuf(table, tableMax, kGarbage);
    fillBuf(localStarts, tableMax, kGarbage);
    fillBuf(bits, n + 1, kGarbage);
    fillBuf(scratch, (size_t)scratchCount, kGarbage);
  }
#if defined(PMPP_BACKEND_METAL)
  std::string runMulti(int r, int c, pmpp::Timing* tm = nullptr) {
    std::string why = multiProblem(r, c);
    if (!why.empty()) return why;
    bool ok = true;
    pmpp::Timing t = pmpp::run([&](void* v) {
      ok = encodeRadixSortMulti(encoderOf(v), pso("radix_local_sort"), pso("radix_scatter_local"), buffer(keys),
                                buffer(vals), buffer(tmpKeys), buffer(tmpVals), buffer(table), buffer(localStarts),
                                buffer(scratch), n, (uint32_t)r, (uint32_t)c);
    });
    if (tm) *tm = t;
    return ok ? "" : "encodeRadixSortMulti returned false, although this configuration fits the device";
  }
  std::string runBaseline(pmpp::Timing* tm = nullptr) {
    bool ok = true;
    pmpp::Timing t = pmpp::run([&](void* v) {
      ok = encodeRadixSortOneBitGlobal(encoderOf(v), buffer(keys), buffer(vals), buffer(tmpKeys), buffer(tmpVals),
                                       buffer(bits), buffer(scratch), n);
    });
    if (tm) *tm = t;
    return ok ? "" : "the provided baseline did not run: " + pmpp::take_error();
  }
#else
  std::string runMulti(int r, int c, pmpp::Timing* tm = nullptr) {
    std::string why = multiProblem(r, c);
    if (!why.empty()) return why;
    pmpp::Timing t = pmpp::run([&](void*) {
      launchRadixSortMulti(u(keys), u(vals), u(tmpKeys), u(tmpVals), u(table), u(localStarts), u(scratch), (int)n, r, c);
    });
    if (tm) *tm = t;
    return "";
  }
  std::string runBaseline(pmpp::Timing* tm = nullptr) {
    pmpp::Timing t = pmpp::run([&](void*) {
      launchRadixSortOneBitGlobal(u(keys), u(vals), u(tmpKeys), u(tmpVals), u(bits), u(scratch), (int)n);
    });
    if (tm) *tm = t;
    return "";
  }
#endif
  // "" when keys/vals equal std::stable_sort's keys and permutation, and no guard slot changed.
  std::string check(const Case& cs) {
    const std::vector<uint32_t> gk = downloadU32(keys, n + kPad), gv = downloadU32(vals, n + kPad);
    const std::vector<uint32_t> tk = downloadU32(tmpKeys, n + kPad), tv = downloadU32(tmpVals, n + kPad);
    const char* names[4] = {"keys", "vals", "tmpKeys", "tmpVals"};
    const std::vector<uint32_t>* arrays[4] = {&gk, &gv, &tk, &tv};
    for (int a = 0; a < 4; ++a)
      for (uint32_t i = n; i < n + kPad; ++i)
        if ((*arrays[a])[i] != kGarbage)
          return std::string(names[a]) + fmt("[%u] was written, but only [0, %u) belongs to the sort (padding of the last tile?)", i, n);
    for (uint32_t i = 0; i < n; ++i) {
      if (gk[i] != cs.refKeys[i]) {
        if (gk[i] == kGarbage) return fmt("keys[%u] is 0xCDCDCDCD, the garbage the driver wrote: no scatter wrote it", i);
        return fmt("keys[%u] = 0x%08x, std::stable_sort has 0x%08x there", i, gk[i], cs.refKeys[i]);
      }
    }
    for (uint32_t i = 0; i < n; ++i) {
      if (gv[i] != cs.refVals[i]) {
        if (gv[i] == kGarbage) return fmt("the keys are right, but vals[%u] was never written", i);
        return fmt("the keys are right, but vals[%u] holds input position %u where std::stable_sort has %u: "
                   "equal keys left their input order, or a value did not move with its key", i, gv[i], cs.refVals[i]);
      }
    }
    return "";
  }
};

// Sorts every (n, pattern) listed with (r, c) (r = 0: the provided baseline). "" if all are right.
struct Sub { uint32_t n; KeyPattern p; };
static std::string runSubs(int r, int c, const std::vector<Sub>& subs) {
  std::map<uint32_t, std::unique_ptr<Problem>> problems;
  for (const Sub& s : subs) {
    auto& prob = problems[s.n];
    if (!prob) prob.reset(new Problem(s.n));
    const Case& cs = getCase(s.p, s.n);
    prob->prepare(cs.keys);
    std::string err = r == 0 ? prob->runBaseline() : prob->runMulti(r, c);
    if (err.empty()) err = pmpp::take_error();
    if (err.empty()) err = prob->check(cs);
    if (!err.empty()) return "n = " + fmtN(s.n) + ", " + keyName(s.p) + ": " + err;
  }
  return "";
}
static std::vector<Sub> crossSubs(const std::vector<uint32_t>& sizes, const std::vector<KeyPattern>& pats) {
  std::vector<Sub> out;
  for (uint32_t n : sizes)
    for (KeyPattern p : pats) out.push_back({n, p});
  return out;
}
static std::string describeSubs(const std::vector<Sub>& subs) {
  std::string sizes, pats;
  std::vector<uint32_t> seenN;
  std::vector<int> seenP;
  for (const Sub& s : subs) {
    if (std::find(seenN.begin(), seenN.end(), s.n) == seenN.end()) { seenN.push_back(s.n); sizes += (sizes.empty() ? "" : ", ") + fmtN(s.n); }
    if (std::find(seenP.begin(), seenP.end(), (int)s.p) == seenP.end()) { seenP.push_back((int)s.p); pats += (pats.empty() ? "" : ", ") + std::string(keyName(s.p)); }
  }
  if (seenP.size() == kKeyPatterns.size()) pats = "8 key patterns";
  return "n = " + sizes + ": " + pats;
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  const bool race = be == "race", emu = be == "emu", small = race || emu;

  // ---- the provided one-bit baseline (not yours: a failure here is a broken provided file)
  {
    std::vector<Sub> subs = small ? std::vector<Sub>{{257, KRandom}, {257, KBelow256}}
                                  : crossSubs({1, 257, 4097, 65537}, kKeyPatterns);
    t.check("provided one-bit baseline, " + describeSubs(subs), [&]() { return runSubs(0, 0, subs); });
  }

  // ---- your sort. On a GPU: every r and c with every size and every key pattern. The CPU
  // emulators pay for every thread of every block of every launch (3 per pass, up to 32 passes
  // per sort), so they run each (r, c) with c = 1 or 2 (except r = 1, c = 2, the slowest there)
  // on n = 1, 257 and 1000 with key patterns that rotate across the configurations, then a few
  // targeted cases.
  const std::vector<int> cs = small ? std::vector<int>{1, 2} : std::vector<int>{1, 2, 4};
  int idx = 0;
  for (int r : kTestRadix) {
    for (int c : cs) {
      if (small && r == 1 && c == 2) continue;
      const std::string label = "r = " + std::to_string(r) + ", c = " + std::to_string(c) + " (" +
                                std::to_string(passesFor(r)) + " passes), ";
      std::string why = multiProblem(r, c);
      if (!why.empty() && why.find("did not build") == std::string::npos) {
        std::printf("  skipped  r = %d, c = %d: %s\n", r, c, why.c_str());
        ++idx;
        continue;
      }
      std::vector<Sub> subs;
      if (small) {
        subs = {{1, KRandom}, {257, kKeyPatterns[idx % 8]}, {1000, kKeyPatterns[(idx + 3) % 8]}};
      } else {
        subs = crossSubs({1, 255, 256, 257, 1000, 4097, 65537, 1u << 20}, kKeyPatterns);
      }
      t.check(label + describeSubs(subs), [&, r, c, subs]() { return runSubs(r, c, subs); });
      ++idx;
    }
  }
  if (small) {
    const std::vector<std::pair<std::pair<int, int>, std::vector<Sub>>> extra = {
        {{4, 2}, crossSubs({600}, kKeyPatterns)},
        {{5, 1}, crossSubs({255, 256}, {KRandom, KReversed})},
        {{8, 2}, crossSubs({255}, {KSameLowByte, KBelow256})},
    };
    for (const auto& e : extra) {
      const int r = e.first.first, c = e.first.second;
      t.check("r = " + std::to_string(r) + ", c = " + std::to_string(c) + " (" + std::to_string(passesFor(r)) +
                  " passes), " + describeSubs(e.second),
              [&, r, c]() { return runSubs(r, c, e.second); });
    }
    if (emu) {
      // 17 blocks of 2^8 buckets: a table of 4352 entries, so the provided scan needs several tiles.
      for (std::pair<int, int> rc : {std::make_pair(8, 1)}) {
        const std::vector<Sub> subs = {{4097, KRandom}};
        t.check("r = " + std::to_string(rc.first) + ", c = " + std::to_string(rc.second) + " (" +
                    std::to_string(passesFor(rc.first)) + " passes), " + describeSubs(subs),
                [&, rc, subs]() { return runSubs(rc.first, rc.second, subs); });
      }
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Best device time of `reps` runs after a warm-up; every run's result is checked first.
// Returns a negative time and sets *err when a run is wrong or cannot run.
static double timeSort(Problem& p, const Case& cs, int reps, int r, int c, std::string* err) {
  double best = 1e30;
  for (int k = 0; k <= reps; ++k) {
    p.prepare(cs.keys);
    pmpp::Timing tm{0, -1};
    std::string e = r == 0 ? p.runBaseline(&tm) : p.runMulti(r, c, &tm);
    if (e.empty()) e = pmpp::take_error();
    if (e.empty()) e = p.check(cs);
    if (!e.empty()) { *err = e; return -1; }
    if (k > 0) best = std::min(best, tm.best());
  }
  return best;
}

static int bench(const pmpp::Args& a) {
  int rc = 0;
  const double mem = deviceMemory();
  // Start at 2^22 pairs and double while the baseline takes under 20 ms, staying within an
  // eighth of the device's memory (about 40 bytes per pair with the table and scratch).
  uint32_t n = 1u << 22;
  double baseS = 0;
  std::string err;
  for (;;) {
    Problem p(n);
    baseS = timeSort(p, getCase(KRandom, n), a.reps, 0, 0, &err);
    if (baseS < 0) { std::printf("the provided baseline failed: %s\n", err.c_str()); return 1; }
    const bool roomToGrow = mem <= 0 || 2.0 * n * 40.0 < mem / 8;
    if (baseS >= 0.02 || !roomToGrow || n >= (1u << 25)) break;
    n *= 2;
  }
  const Case& cs = getCase(KRandom, n);
  double cpu = 1e30;
  {
    std::vector<uint32_t> rk, rv;
    for (int k = 0; k <= a.reps; ++k) {
      const auto t0 = std::chrono::steady_clock::now();
      cpuStableSort(cs.keys, rk, rv);
      const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
      if (k > 0) cpu = std::min(cpu, s);
    }
  }
  std::printf("n = %s pairs of random 32-bit keys and their positions, best of %d runs after a warm-up.\n",
              fmtN(n).c_str(), a.reps);
  std::printf("Time: device time of the command buffer on Metal; wall time around the launches and the final\n"
              "synchronisation on CUDA and HIP. CPU: std::stable_sort on one CPU thread (a host measurement).\n");
  std::printf("\"Counted GB/s\" is counted, not measured traffic: each pass reads and writes every pair twice\n"
              "(32 bytes per pair per pass; the table and the scan are left out).\n\n");
  std::printf("  %-14s %3s %3s %7s %10s %10s %13s %12s %10s\n", "kernel", "r", "c", "passes", "ms", "Mkeys/s",
              "counted GB/s", "vs baseline", "vs CPU");
  std::printf("  %-14s %3s %3s %7d %10.3f %10.1f %13s %12s %9.2fx\n", "one-bit global", "1", "-", 32, baseS * 1e3,
              n / baseS / 1e6, "", "1.00x", cpu / baseS);
  pmpp::bench_row(a, "onebit_global").n("n", n).n("seconds", baseS).n("mkeys_per_s", n / baseS / 1e6).emit();

  Problem p(n);
  const std::vector<std::pair<int, int>> configs = {{1, 1}, {2, 1}, {4, 1}, {6, 1}, {8, 1}, {4, 2},
                                                     {4, 4}, {4, 8}, {8, 2}, {8, 4}, {8, 8}};
  for (const auto& rcfg : configs) {
    const int r = rcfg.first, c = rcfg.second;
    std::string why = multiProblem(r, c);
    if (!why.empty()) {
      std::printf("  %-14s %3d %3d  skipped: %s\n", "local", r, c, why.c_str());
      continue;
    }
    const double s = timeSort(p, cs, a.reps, r, c, &err);
    if (s < 0) {
      std::printf("  %-14s %3d %3d  not timed, the result is wrong: %s\n", "local", r, c, err.c_str());
      rc = 1;
      continue;
    }
    const int passes = passesFor(r);
    const double counted = 32.0 * passes * n / s / 1e9;
    std::printf("  %-14s %3d %3d %7d %10.3f %10.1f %13.1f %11.2fx %9.2fx\n", "local", r, c, passes, s * 1e3,
                n / s / 1e6, counted, baseS / s, cpu / s);
    pmpp::bench_row(a, "local").n("n", n).n("radix_bits", r).n("keys_per_thread", c).n("passes", passes)
        .n("seconds", s).n("mkeys_per_s", n / s / 1e6).n("counted_gbps", counted).emit();
  }
  std::printf("  %-14s %3s %3s %7s %10.3f %10.1f\n", "CPU stable", "", "", "", cpu * 1e3, n / cpu / 1e6);
  pmpp::bench_row(a, "cpu_stable_sort").n("n", n).n("seconds", cpu).n("mkeys_per_s", n / cpu / 1e6).emit();
  std::printf("\n\"vs baseline\" and \"vs CPU\" are speed-ups (their time divided by this row's). These are\n"
              "measurements of this machine at this moment, not targets.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
