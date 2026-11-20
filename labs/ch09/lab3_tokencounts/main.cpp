// Lab 9.3 driver: tests and benchmarks the token counts behind frequency and presence
// penalties (milestone M5): your countTokens, appendTokens and applyPenalties, and the
// provided count_pool (Fig. 9.9's design, in provided/) as the comparator.
// You do not need to edit this file; your work goes in work/.
//
// The CPU reference lives here. Counts must match exactly. Penalised logits must match
// within a relative tolerance, because Metal does not default to strict IEEE arithmetic
// (notes/today/ch05-facts.md C7); with both penalties 0 they must not change at all.
// Device buffers start as garbage (Metal: 0xA5 bytes), so this driver zeroes `counts`
// before every countTokens, as the contract says the caller does.
#include <chrono>
#include <cstring>
#include <map>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- entry points implemented in <variant>/ and provided/ ----
// Each run* returns "" when the kernel was launched, else why it was not.
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeCountTokens(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> tokens,
                       id<MTLBuffer> counts, uint32_t batch, uint32_t len, uint32_t vocab);
void encodeAppendTokens(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> newTokens,
                        id<MTLBuffer> counts, uint32_t batch, uint32_t vocab);
void encodeApplyPenalties(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> logits,
                          id<MTLBuffer> counts, uint32_t batch, uint32_t vocab,
                          float frequencyPenalty, float presencePenalty);
uint32_t countPoolBlocksPerRow(uint32_t len);
bool encodeCountPool(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> zero,
                     id<MTLComputePipelineState> count, id<MTLBuffer> tokens, id<MTLBuffer> counts,
                     id<MTLBuffer> pool, uint32_t batch, uint32_t len, uint32_t vocab);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
static id<MTLComputePipelineState> poolPso(const char* fn) {
  return pmpp::metal::pipeline(dirOf(dirOf(PMPP_METAL_SOURCE)) + "/provided/count_pool.metal", fn);
}
static std::string notBuilt(const char* fn) {
  return std::string("the Metal kernel ") + fn + " did not build: " + pmpp::take_error();
}
static std::string runCount(Buf* tokens, Buf* counts, int batch, int len, int vocab, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = pso("count_tokens");
  if (!p) return notBuilt("count_tokens");
  pmpp::Timing t = pmpp::run([&](void* v) {
    encodeCountTokens(((pmpp::metal::Pass*)v)->encoder, p, buffer(tokens), buffer(counts), batch, len, vocab);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runAppend(Buf* newTokens, Buf* counts, int batch, int vocab, pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = pso("append_tokens");
  if (!p) return notBuilt("append_tokens");
  pmpp::Timing t = pmpp::run([&](void* v) {
    encodeAppendTokens(((pmpp::metal::Pass*)v)->encoder, p, buffer(newTokens), buffer(counts), batch, vocab);
  });
  if (tm) *tm = t;
  return "";
}
static std::string runPenalties(Buf* logits, Buf* counts, int batch, int vocab, float freq, float pres,
                                pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> p = pso("apply_penalties");
  if (!p) return notBuilt("apply_penalties");
  pmpp::Timing t = pmpp::run([&](void* v) {
    encodeApplyPenalties(((pmpp::metal::Pass*)v)->encoder, p, buffer(logits), buffer(counts), batch, vocab, freq, pres);
  });
  if (tm) *tm = t;
  return "";
}
static long long poolBlocksPerRow(int len) { return countPoolBlocksPerRow((uint32_t)len); }
static std::string runPool(Buf* tokens, Buf* counts, Buf* pool, int batch, int len, int vocab,
                           pmpp::Timing* tm = nullptr) {
  id<MTLComputePipelineState> z = poolPso("zero_pool"), c = poolPso("count_pool");
  if (!z || !c) return notBuilt("zero_pool or count_pool (provided)");
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    ok = encodeCountPool(((pmpp::metal::Pass*)v)->encoder, z, c, buffer(tokens), buffer(counts), buffer(pool),
                         batch, len, vocab);
  });
  if (tm) *tm = t;
  return ok ? "" : "the provided count_pool pipelines cannot run 256 threads per threadgroup here";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* kOnChip = "maxThreadgroupMemoryLength";
#else
void launchCountTokens(const int* tokens, unsigned int* counts, int batch, int len, int vocab);
void launchAppendTokens(const int* newTokens, unsigned int* counts, int batch, int vocab);
void launchApplyPenalties(float* logits, const unsigned int* counts, int batch, int vocab,
                          float frequencyPenalty, float presencePenalty);
int countPoolBlocksPerRow(int len);
void launchCountPool(const int* tokens, unsigned int* counts, unsigned int* pool, int batch, int len, int vocab);

static int* ids(Buf* b) { return (int*)pmpp::raw(b); }
static unsigned int* uints(Buf* b) { return (unsigned int*)pmpp::raw(b); }
static std::string runCount(Buf* tokens, Buf* counts, int batch, int len, int vocab, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) { launchCountTokens(ids(tokens), uints(counts), batch, len, vocab); });
  if (tm) *tm = t;
  return "";
}
static std::string runAppend(Buf* newTokens, Buf* counts, int batch, int vocab, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) { launchAppendTokens(ids(newTokens), uints(counts), batch, vocab); });
  if (tm) *tm = t;
  return "";
}
static std::string runPenalties(Buf* logits, Buf* counts, int batch, int vocab, float freq, float pres,
                                pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    launchApplyPenalties((float*)pmpp::raw(logits), uints(counts), batch, vocab, freq, pres);
  });
  if (tm) *tm = t;
  return "";
}
static long long poolBlocksPerRow(int len) { return countPoolBlocksPerRow(len); }
static std::string runPool(Buf* tokens, Buf* counts, Buf* pool, int batch, int len, int vocab,
                           pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) { launchCountPool(ids(tokens), uints(counts), uints(pool), batch, len, vocab); });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* kOnChip = "sharedMemPerBlock";
#endif

// ---------------------------------------------------------------- data (seeded)
enum Dist { Uniform, Zipf, Repeated, Boundary };
static const Dist kDists[4] = {Uniform, Zipf, Repeated, Boundary};
// uniform: ids uniform over the vocabulary; zipf: a few ids take most of the row;
// repeated: one id fills the row (every thread hits the same bin); boundary: rows that
// hold ids 0 and vocab-1, which catch a wrong row offset.
static const char* distName(Dist d) {
  return d == Uniform ? "uniform" : d == Zipf ? "zipf" : d == Repeated ? "repeated" : "boundary";
}

struct Rng {  // xorshift32: the same numbers on every platform
  uint32_t x;
  explicit Rng(uint32_t seed) : x(seed * 2654435761u + 1u) { if (!x) x = 1; }
  uint32_t next() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }
  int below(int n) { return (int)(((uint64_t)next() * (uint32_t)n) >> 32); }
  double unit() { return (next() >> 8) * (1.0 / 16777216.0); }
};

// Zipf with exponent 1.1 over ranks, sampled through its cumulative distribution. The
// ranks are then spread over the vocabulary (7919 is prime and divides none of the
// vocabulary sizes used), so the popular ids are not all small numbers.
static int zipfId(Rng& g, int vocab) {
  static std::map<int, std::vector<double>> cache;
  std::vector<double>& cdf = cache[vocab];
  if (cdf.empty()) {
    cdf.resize((size_t)vocab);
    double s = 0;
    for (int r = 0; r < vocab; ++r) { s += std::pow(r + 1.0, -1.1); cdf[(size_t)r] = s; }
    for (double& v : cdf) v /= s;
  }
  size_t rank = (size_t)(std::lower_bound(cdf.begin(), cdf.end(), g.unit()) - cdf.begin());
  if (rank >= (size_t)vocab) rank = (size_t)vocab - 1;
  return (int)(((uint64_t)rank * 7919u + 13u) % (uint64_t)vocab);
}
static int repeatedId(int row, int vocab) { return (int)(((uint64_t)row * 7919u + 4321u) % (uint64_t)vocab); }

struct Shape { int batch, len, vocab; };
static std::string shapeName(const Shape& s) {
  char b[96];
  std::snprintf(b, sizeof b, "batch %d, len %d, vocab %d", s.batch, s.len, s.vocab);
  return b;
}

static std::vector<int> makeTokens(Dist d, const Shape& s, uint32_t seed) {
  std::vector<int> t((size_t)s.batch * s.len);
  for (int r = 0; r < s.batch; ++r) {
    Rng g(seed + 977u * (uint32_t)r);
    int* row = &t[(size_t)r * s.len];
    for (int i = 0; i < s.len; ++i) {
      row[i] = d == Zipf ? zipfId(g, s.vocab) : d == Repeated ? repeatedId(r, s.vocab) : g.below(s.vocab);
    }
    if (d == Boundary) {  // the first and last id of the vocabulary, at both ends of the row
      const int lo = 0, hi = s.vocab - 1;
      row[0] = (r % 2) ? hi : lo;
      if (s.len > 1) row[s.len - 1] = (r % 2) ? lo : hi;
      if (s.len > 2) row[s.len / 2] = hi;
    }
  }
  return t;
}
static std::vector<int> makeNewTokens(Dist d, const Shape& s, uint32_t seed) {
  std::vector<int> t((size_t)s.batch);
  Rng g(seed ^ 0x5bd1e995u);
  for (int r = 0; r < s.batch; ++r) {
    t[(size_t)r] = d == Uniform ? g.below(s.vocab)
                 : d == Zipf    ? zipfId(g, s.vocab)
                 : d == Repeated ? repeatedId(r, s.vocab)
                                 : ((r % 2) ? 0 : s.vocab - 1);
  }
  return t;
}

// ---------------------------------------------------------------- CPU reference
// counts[r][t] = occurrences of t among row r's len tokens (plus newTokens[r], if given).
static std::vector<uint32_t> cpuCounts(const std::vector<int>& tokens, const Shape& s,
                                       const std::vector<int>* newTokens = nullptr) {
  std::vector<uint32_t> c((size_t)s.batch * s.vocab, 0);
  for (int r = 0; r < s.batch; ++r) {
    for (int i = 0; i < s.len; ++i) c[(size_t)r * s.vocab + tokens[(size_t)r * s.len + i]]++;
    if (newTokens) c[(size_t)r * s.vocab + (*newTokens)[(size_t)r]]++;
  }
  return c;
}
// llama.cpp's frequency and presence penalty (notes/today/ch09-facts.md D4), in double.
// `mag` receives the size of the terms, which scales the tolerance.
static void cpuPenalties(const std::vector<float>& logits, const std::vector<uint32_t>& counts, float freq,
                         float pres, std::vector<float>& ref, std::vector<float>& mag) {
  ref.resize(logits.size()); mag.resize(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    const double c = counts[i], sub = c * freq + (counts[i] > 0 ? (double)pres : 0.0);
    ref[i] = (float)((double)logits[i] - sub);
    mag[i] = (float)(std::fabs((double)logits[i]) + std::fabs(sub));
  }
}

static std::string compareCounts(const std::vector<uint32_t>& got, const std::vector<uint32_t>& ref, int vocab) {
  size_t bad = 0, first = 0;
  unsigned long long gotSum = 0, refSum = 0;
  for (size_t i = 0; i < ref.size(); ++i) {
    gotSum += got[i]; refSum += ref[i];
    if (got[i] != ref[i]) { if (bad == 0) first = i; ++bad; }
  }
  if (bad == 0) return "";
  char b[300];
  std::snprintf(b, sizeof b, "%zu wrong counts; first at row %zu, id %zu: got %u, expected %u "
                "(all counts add up to %llu, the reference to %llu)",
                bad, first / (size_t)vocab, first % (size_t)vocab, got[first], ref[first], gotSum, refSum);
  return b;
}

// ---------------------------------------------------------------- one problem on the device
struct Problem {
  Shape s;
  std::vector<int> tokens, newTokens;
  Buf *T, *N, *C;
  Problem(const Shape& sh, Dist d, uint32_t seed) : s(sh) {
    tokens = makeTokens(d, s, seed);
    newTokens = makeNewTokens(d, s, seed);
    T = pmpp::alloc(tokens.size() * 4);
    N = pmpp::alloc(newTokens.size() * 4);
    C = pmpp::alloc(countsSize() * 4);
    pmpp::upload(T, tokens.data(), tokens.size() * 4);
    pmpp::upload(N, newTokens.data(), newTokens.size() * 4);
  }
  Problem(const Problem&) = delete;
  ~Problem() { pmpp::release(T); pmpp::release(N); pmpp::release(C); }
  size_t countsSize() const { return (size_t)s.batch * s.vocab; }
  void setCounts(const std::vector<uint32_t>& c) { pmpp::upload(C, c.data(), c.size() * 4); }
  void zeroCounts() { setCounts(std::vector<uint32_t>(countsSize(), 0)); }
  std::vector<uint32_t> counts() {
    std::vector<uint32_t> c(countsSize());
    pmpp::download(C, c.data(), c.size() * 4);
    return c;
  }
};

// ---------------------------------------------------------------- tests
static std::vector<Shape> testShapes() {
  std::vector<Shape> s = {{1, 1, 1000}, {3, 100, 1000}, {2, 64, 50257}};
  if (!pmpp::is_emu()) {
    const std::vector<Shape> more = {{1, 1, 50257}, {3, 1000, 50257}, {2, 4096, 151936},
                                     {4, 777, 201088}, {8, 2048, 151936}};
    s.insert(s.end(), more.begin(), more.end());
  }
  return s;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  uint32_t seed = 9000;
  for (const Shape& s : testShapes()) {
    for (Dist d : kDists) {
      ++seed;
      const std::string what = shapeName(s) + ", " + distName(d);
      t.check("countTokens, " + what, [&]() -> std::string {
        Problem p(s, d, seed);
        p.zeroCounts();
        std::string e = runCount(p.T, p.C, s.batch, s.len, s.vocab);
        if (!e.empty()) return e;
        return compareCounts(p.counts(), cpuCounts(p.tokens, s), s.vocab);
      });
      // One decode step: countTokens over len tokens, then appendTokens with one more per
      // row, must equal counting all len + 1 tokens.
      t.check("countTokens then appendTokens, " + what, [&]() -> std::string {
        Problem p(s, d, seed);
        p.zeroCounts();
        std::string e = runCount(p.T, p.C, s.batch, s.len, s.vocab);
        if (!e.empty()) return e;
        const std::vector<uint32_t> before = cpuCounts(p.tokens, s);
        std::string prefix;
        if (!compareCounts(p.counts(), before, s.vocab).empty()) {
          prefix = "countTokens is wrong (see its case), so appendTokens was run on the reference counts: ";
          p.setCounts(before);
        }
        e = runAppend(p.N, p.C, s.batch, s.vocab);
        if (!e.empty()) return prefix + e;
        const std::string r = compareCounts(p.counts(), cpuCounts(p.tokens, s, &p.newTokens), s.vocab);
        if (r.empty()) return prefix.empty() ? "" : prefix + "appendTokens itself is right";
        return prefix + "appendTokens: " + r;
      });
    }
    ++seed;
    // Penalties on the counts of Zipf-like rows: many zeros, some ones, a few large counts.
    t.check("applyPenalties (0.5, 0.25) and (0, 0), " + shapeName(s), [&]() -> std::string {
      Problem p(s, Zipf, seed);
      const std::vector<uint32_t> counts = cpuCounts(p.tokens, s);
      p.setCounts(counts);
      std::vector<float> logits = pmpp::random_floats(counts.size(), seed);
      for (float& x : logits) x *= 8.0f;
      Buf* L = pmpp::alloc(logits.size() * 4);
      std::string err;
      std::vector<float> got(logits.size()), ref, mag;
      pmpp::upload(L, logits.data(), logits.size() * 4);
      err = runPenalties(L, p.C, s.batch, s.vocab, 0.5f, 0.25f);
      if (err.empty()) {
        pmpp::download(L, got.data(), got.size() * 4);
        cpuPenalties(logits, counts, 0.5f, 0.25f, ref, mag);
        err = pmpp::describe(pmpp::compare(got.data(), ref.data(), mag.data(), got.size(), 1e-5), s.vocab);
        if (!err.empty()) err = "penalties (0.5, 0.25): " + err + " (col = token id)";
      }
      if (err.empty()) {
        pmpp::upload(L, logits.data(), logits.size() * 4);
        err = runPenalties(L, p.C, s.batch, s.vocab, 0.0f, 0.0f);
        if (err.empty()) {
          pmpp::download(L, got.data(), got.size() * 4);
          size_t changed = 0, first = 0;
          for (size_t i = 0; i < got.size(); ++i)
            if (std::memcmp(&got[i], &logits[i], 4) != 0) { if (changed == 0) first = i; ++changed; }
          if (changed) {
            char b[240];
            std::snprintf(b, sizeof b, "penalties (0, 0) must leave the logits unchanged: %zu changed; "
                          "first at row %zu, id %zu: %.9g became %.9g", changed, first / (size_t)s.vocab,
                          first % (size_t)s.vocab, logits[first], got[first]);
            err = b;
          }
        }
      }
      pmpp::release(L);
      return err;
    });
    // The provided comparator, so a broken copy of provided/ shows.
    t.check("provided count_pool, " + shapeName(s) + ", zipf", [&]() -> std::string {
      Problem p(s, Zipf, seed);
      Buf* pool = pmpp::alloc((size_t)s.batch * poolBlocksPerRow(s.len) * s.vocab * 4);
      p.zeroCounts();
      std::string e = runPool(p.T, p.C, pool, s.batch, s.len, s.vocab);
      if (e.empty()) e = compareCounts(p.counts(), cpuCounts(p.tokens, s), s.vocab);
      pmpp::release(pool);
      return e;
    });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
static void row(const pmpp::Args& a, const char* name, Dist d, const Shape& s, double sec, const char* rateKey,
                double rate, double blocksPerRow = -1, double poolBytes = -1) {
  pmpp::Json j = pmpp::bench_row(a, name);
  j.s("data", distName(d)).n("batch", s.batch).n("len", s.len).n("vocab", s.vocab).n("seconds", sec).n(rateKey, rate);
  if (blocksPerRow >= 0) j.n("blocks_per_row", blocksPerRow).n("pool_bytes", poolBytes);
  j.emit();
}

static int bench(const pmpp::Args& a) {
  const Shape s = {8, 4096, 151936};
  const int reps = std::max(a.reps, 10);
  const double countsBytes = (double)s.batch * s.vocab * 4;
  const long long bpr = poolBlocksPerRow(s.len);
  const double poolBytes = (double)s.batch * bpr * s.vocab * 4;
  const double mem = deviceMemory();
  // The pool is the one large allocation. Run it only if it takes at most a quarter of
  // the memory this device reports.
  const bool poolFits = mem <= 0 || poolBytes <= mem / 4;

  std::printf("%s: %d tokens to count, %.1f MB of counts, %.1f MB of logits; best of %d runs\n",
              shapeName(s).c_str(), s.batch * s.len, countsBytes / 1e6, countsBytes / 1e6, reps);
  std::printf("one private copy of this vocabulary: %.1f KiB of uint32; this device reports %s = %.1f KiB\n",
              s.vocab * 4.0 / 1024, kOnChip, pmpp::device_prop(kOnChip, 0) / 1024);
  std::printf("count_pool (provided): %lld blocks per row, so %.1f MB of private copies to zero and commit\n",
              bpr, poolBytes / 1e6);
  if (!poolFits) std::printf("  skipped: that is more than a quarter of the %.0f MB this device reports\n", mem / 1e6);

  // Untimed work first, so the first kernel timed does not also pay for waking the device.
  {
    Problem p(s, Uniform, 1);
    p.zeroCounts();
    Buf* L = pmpp::alloc((size_t)s.batch * s.vocab * 4);
    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < 0.25)
      runPenalties(L, p.C, s.batch, s.vocab, 0.0f, 0.0f);
    pmpp::release(L);
    pmpp::take_error();
  }

  std::printf("  %-8s  %-22s %11s %13s %13s\n", "data", "kernel", "time ms", "tokens/s", "elements/s");
  for (Dist d : {Uniform, Zipf}) {
    Problem p(s, d, d == Uniform ? 41 : 42);
    const std::vector<uint32_t> ref = cpuCounts(p.tokens, s);
    const double tokens = (double)s.batch * s.len, elements = (double)s.batch * s.vocab;
    pmpp::Timing tm{0, 0};

    // count (yours): the caller zeroes the counts before each run (outside the timing).
    std::string err;
    const double countS = pmpp::time_it(reps, [&] {
      p.zeroCounts();
      err = runCount(p.T, p.C, s.batch, s.len, s.vocab, &tm);
      return tm;
    }).min_s;
    std::string bad = err.empty() ? compareCounts(p.counts(), ref, s.vocab) : err;
    if (bad.empty()) bad = pmpp::take_error();
    if (!bad.empty()) { std::printf("  %-8s  count: wrong, not timed: %s\n", distName(d), bad.c_str()); return 1; }
    std::printf("  %-8s  %-22s %11.4f %13.3g\n", distName(d), "count (yours)", countS * 1e3, tokens / countS);
    row(a, "count", d, s, countS, "tokens_per_s", tokens / countS);

    // count_pool (provided): Fig. 9.9's per-block private copies, zeroing included.
    double poolS = -1;
    if (poolFits) {
      Buf* pool = pmpp::alloc((size_t)poolBytes);
      poolS = pmpp::time_it(reps, [&] {
        p.zeroCounts();
        err = runPool(p.T, p.C, pool, s.batch, s.len, s.vocab, &tm);
        return tm;
      }).min_s;
      bad = err.empty() ? compareCounts(p.counts(), ref, s.vocab) : err;
      if (bad.empty()) bad = pmpp::take_error();
      pmpp::release(pool);
      if (!bad.empty()) {
        std::printf("  %-8s  count_pool: wrong, not reported: %s\n", distName(d), bad.c_str());
        poolS = -1;
      } else {
        std::printf("  %-8s  %-22s %11.4f %13.3g\n", distName(d), "count_pool (provided)", poolS * 1e3, tokens / poolS);
        row(a, "count_pool", d, s, poolS, "tokens_per_s", tokens / poolS, (double)bpr, poolBytes);
      }
    }

    // append (yours): one decode step, one new token per row, on the counts just made.
    p.setCounts(ref);
    const double appendS = pmpp::time_it(reps, [&] {
      err = runAppend(p.N, p.C, s.batch, s.vocab, &tm);
      return tm;
    }).min_s;
    if (!err.empty()) { std::printf("  append failed: %s\n", err.c_str()); return 1; }
    std::printf("  %-8s  %-22s %11.4f %13.3g\n", distName(d), "append (yours)", appendS * 1e3, s.batch / appendS);
    row(a, "append", d, s, appendS, "tokens_per_s", s.batch / appendS);

    // penalties (yours): every logit of the batch, on these counts.
    p.setCounts(ref);
    Buf* L = pmpp::alloc((size_t)elements * 4);
    std::vector<float> logits = pmpp::random_floats((size_t)elements, 43);
    pmpp::upload(L, logits.data(), logits.size() * 4);
    const double penS = pmpp::time_it(reps, [&] {
      err = runPenalties(L, p.C, s.batch, s.vocab, 0.5f, 0.25f, &tm);
      return tm;
    }).min_s;
    pmpp::release(L);
    if (!err.empty()) { std::printf("  penalties failed: %s\n", err.c_str()); return 1; }
    std::printf("  %-8s  %-22s %11.4f %13s %13.3g   (%.1f GB/s at 12 bytes per logit)\n", distName(d),
                "penalties (yours)", penS * 1e3, "", elements / penS, 12.0 * elements / penS / 1e9);
    row(a, "penalties", d, s, penS, "elements_per_s", elements / penS);

    std::printf("  %-8s  measured: recounting the whole context took %.1fx one append step",
                distName(d), countS / appendS);
    if (poolS > 0) std::printf("; count_pool took %.1fx count", poolS / countS);
    std::printf("\n");
  }
  std::printf("These are measurements of this machine at this size, not properties of the kernels.\n");
  return pmpp::take_error().empty() ? 0 : 1;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
