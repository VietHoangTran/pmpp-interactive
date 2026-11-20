// Lab 13.1 driver: tests and benchmarks the co-rank merge kernels.
// You do not need to edit this file; your work goes in work/.
//
// Every comparison is exact. A merge copies elements and computes nothing, so there is no
// rounding and no tolerance. Each element's value records where it came from (A[i] holds
// i, B[j] holds 1,000,000,000 + j), and the reference is std::merge comparing keys only,
// which the C++ standard guarantees to be stable with the first range winning ties
// (notes/today/ch13-facts.md A6). An exact match of C[0 .. total) therefore also checks
// that equal keys came out in the right order.
//
// C starts filled with a sentinel {INT_MIN, -1} and is longer than m + n, so an output
// your kernel never wrote, and anything written at or past C[total], are both reported.
#include <climits>
#include <cstring>
#include <memory>
#include <set>

#include "merge.h"
#include "pmpp/harness.hpp"

using pmpp::Buf;

static const int kFromB = 1000000000;      // B[j] has value kFromB + j
static const KV kSentinel = {INT_MIN, -1};
static const int kGuard = 64;              // C has m + n + kGuard elements

enum class Kernel { Basic, Tiled };
static const char* kernelName(Kernel k) { return k == Kernel::Basic ? "basic" : "tiled"; }

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeCoRank(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A, uint32_t m,
                  id<MTLBuffer> B, uint32_t n, id<MTLBuffer> out);
bool encodeMergeBasic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> A, uint32_t m,
                      id<MTLBuffer> B, uint32_t n, id<MTLBuffer> C, uint32_t total, uint32_t perThread);
bool encodeMergeTiled(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> partition,
                      id<MTLComputePipelineState> tiled, id<MTLBuffer> A, uint32_t m, id<MTLBuffer> B, uint32_t n,
                      id<MTLBuffer> C, uint32_t total, uint32_t perThread, id<MTLBuffer> bounds);
using pmpp::metal::buffer;

static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
static std::string notBuilt() { return "the Metal kernel did not build: " + pmpp::take_error(); }

// Why this device cannot run the kernel at this perThread, or "" if it can. This is the
// driver's own copy of the rule a correct launch.mm applies. A pipeline that does not
// build is not a limit: the run reports it.
static std::string limitProblem(Kernel k, int perThread) {
  const std::vector<const char*> fns =
      k == Kernel::Basic ? std::vector<const char*>{"merge_basic"} : std::vector<const char*>{"merge_partition", "merge_tiled"};
  char b[300];
  for (const char* fn : fns) {
    id<MTLComputePipelineState> p = pso(fn);
    if (!p) { pmpp::take_error(); return ""; }
    if (p.maxTotalThreadsPerThreadgroup < (NSUInteger)MERGE_BLOCK) {
      std::snprintf(b, sizeof b, "%s runs at most %lu threads per threadgroup here, fewer than %d", fn,
                    (unsigned long)p.maxTotalThreadsPerThreadgroup, MERGE_BLOCK);
      return b;
    }
  }
  if (k == Kernel::Tiled) {
    id<MTLComputePipelineState> p = pso("merge_tiled");
    const unsigned long need = (unsigned long)MERGE_BLOCK * perThread * sizeof(KV) + (unsigned long)p.staticThreadgroupMemoryLength;
    const unsigned long have = (unsigned long)pmpp::metal::device().maxThreadgroupMemoryLength;
    if (need > have) {
      std::snprintf(b, sizeof b, "a tile of %d elements needs %lu bytes of threadgroup memory; this device allows %lu",
                    MERGE_BLOCK * perThread, need, have);
      return b;
    }
  }
  return "";
}

static std::string runCoRank(Buf* A, int m, Buf* B, int n, Buf* out) {
  id<MTLComputePipelineState> p = pso("co_rank_test");
  if (!p) return notBuilt();
  pmpp::run([&](void* v) {
    encodeCoRank(((pmpp::metal::Pass*)v)->encoder, p, buffer(A), (uint32_t)m, buffer(B), (uint32_t)n, buffer(out));
  });
  return "";
}

// Empty string when the kernel ran; otherwise why it did not.
static std::string runMerge(Kernel k, Buf* A, int m, Buf* B, int n, Buf* C, int total, int perThread, Buf* bounds,
                            pmpp::Timing* tm = nullptr) {
  bool ok = true;
  pmpp::Timing t{0, 0};
  if (k == Kernel::Basic) {
    id<MTLComputePipelineState> p = pso("merge_basic");
    if (!p) return notBuilt();
    t = pmpp::run([&](void* v) {
      ok = encodeMergeBasic(((pmpp::metal::Pass*)v)->encoder, p, buffer(A), (uint32_t)m, buffer(B), (uint32_t)n,
                            buffer(C), (uint32_t)total, (uint32_t)perThread);
    });
  } else {
    id<MTLComputePipelineState> pp = pso("merge_partition"), pt = pso("merge_tiled");
    if (!pp || !pt) return notBuilt();
    t = pmpp::run([&](void* v) {
      ok = encodeMergeTiled(((pmpp::metal::Pass*)v)->encoder, pp, pt, buffer(A), (uint32_t)m, buffer(B), (uint32_t)n,
                            buffer(C), (uint32_t)total, (uint32_t)perThread, buffer(bounds));
    });
  }
  if (tm) *tm = t;
  if (!ok) return "your launch.mm refused perThread " + std::to_string(perThread) + ", which this device can run";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
static KV* kv(Buf* b) { return (KV*)pmpp::raw(b); }

static std::string limitProblem(Kernel k, int perThread) {
  char b[300];
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (most < MERGE_BLOCK) {
    std::snprintf(b, sizeof b, "this device runs at most %d threads per block, fewer than %d", most, MERGE_BLOCK);
    return b;
  }
  if (k == Kernel::Tiled) {
    const double need = (double)MERGE_BLOCK * perThread * sizeof(KV);
    const double have = pmpp::device_prop("sharedMemPerBlock", 0);
    if (need > have) {
      std::snprintf(b, sizeof b, "a tile of %d elements needs %.0f bytes of shared memory; this device allows %.0f per block",
                    MERGE_BLOCK * perThread, need, have);
      return b;
    }
  }
  return "";
}

static std::string runCoRank(Buf* A, int m, Buf* B, int n, Buf* out) {
  pmpp::run([&](void*) { launchCoRank(kv(A), m, kv(B), n, (int*)pmpp::raw(out)); });
  return "";
}

static std::string runMerge(Kernel k, Buf* A, int m, Buf* B, int n, Buf* C, int total, int perThread, Buf* bounds,
                            pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    if (k == Kernel::Basic) launchMergeBasic(kv(A), m, kv(B), n, kv(C), total, perThread);
    else launchMergeTiled(kv(A), m, kv(B), n, kv(C), total, perThread, (int*)pmpp::raw(bounds));
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- data
struct Rng {
  uint32_t x;
  explicit Rng(uint32_t seed) : x(seed * 2654435761u + 1u) {}
  uint32_t next() { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }   // xorshift32
};

// `count` keys uniform in [0, range), sorted.
static std::vector<int> smallKeys(size_t count, int range, Rng& r) {
  std::vector<int> v(count);
  for (int& k : v) k = (int)(r.next() % (uint32_t)range);
  std::sort(v.begin(), v.end());
  return v;
}
// `count` keys uniform over the non-negative 31-bit integers, sorted.
static std::vector<int> wideKeys(size_t count, Rng& r) {
  std::vector<int> v(count);
  for (int& k : v) k = (int)(r.next() >> 1);
  std::sort(v.begin(), v.end());
  return v;
}
// m + n different keys in increasing order, dealt to A and B at random.
static void distinctKeys(int m, int n, Rng& r, std::vector<int>* a, std::vector<int>* b) {
  std::vector<char> toA((size_t)(m + n), 0);
  std::fill(toA.begin(), toA.begin() + m, 1);
  for (size_t i = toA.size(); i > 1; --i) std::swap(toA[i - 1], toA[r.next() % i]);
  int key = 0;
  for (char t : toA) { key += 1 + (int)(r.next() % 3); (t ? a : b)->push_back(key); }
}

struct Case {
  std::string name;
  std::vector<KV> A, B;
  int m() const { return (int)A.size(); }
  int n() const { return (int)B.size(); }
};
static Case makeCase(const std::string& name, const std::vector<int>& a, const std::vector<int>& b) {
  Case c{name, {}, {}};
  for (size_t i = 0; i < a.size(); ++i) c.A.push_back(KV{a[i], (int)i});
  for (size_t j = 0; j < b.size(); ++j) c.B.push_back(KV{b[j], kFromB + (int)j});
  return c;
}
static std::vector<int> consecutive(int from, int count) {   // from, from + 1, ...
  std::vector<int> v((size_t)count);
  for (int i = 0; i < count; ++i) v[i] = from + i;
  return v;
}

static bool byKey(const KV& x, const KV& y) { return x.key < y.key; }
static bool same(const KV& x, const KV& y) { return x.key == y.key && x.val == y.val; }

// The stable merge of A and B (ties to A), and the same with ties to B.
static std::vector<KV> reference(const Case& c) {
  std::vector<KV> out(c.A.size() + c.B.size());
  std::merge(c.A.begin(), c.A.end(), c.B.begin(), c.B.end(), out.begin(), byKey);
  return out;
}
static std::vector<KV> tiesToB(const Case& c) {
  std::vector<KV> out(c.A.size() + c.B.size());
  std::merge(c.B.begin(), c.B.end(), c.A.begin(), c.A.end(), out.begin(), byKey);
  return out;
}
// co[k] = how many of merged[0 .. k) came from A, for k = 0 .. size.
static std::vector<int> coRanks(const std::vector<KV>& merged) {
  std::vector<int> co(merged.size() + 1, 0);
  for (size_t k = 0; k < merged.size(); ++k) co[k + 1] = co[k] + (merged[k].val < kFromB ? 1 : 0);
  return co;
}

// Which input element e is: 'A', 'B', or 0 when it is neither.
static char origin(const KV& e, const Case& c) {
  if (e.val >= 0 && e.val < c.m() && same(c.A[e.val], e)) return 'A';
  if (e.val >= kFromB && e.val - kFromB < c.n() && same(c.B[e.val - kFromB], e)) return 'B';
  return 0;
}
// "A[3]", "B[1]", or what the element is when it is neither.
static std::string source(const KV& e, const Case& c) {
  if (same(e, kSentinel)) return "the sentinel";
  if (origin(e, c) == 'A') return "A[" + std::to_string(e.val) + "]";
  if (origin(e, c) == 'B') return "B[" + std::to_string(e.val - kFromB) + "]";
  return "{key " + std::to_string(e.key) + ", value " + std::to_string(e.val) + "}, which is no input element";
}
static std::string keyList(const std::vector<KV>& v) {
  std::string s;
  for (const KV& e : v) s += (s.empty() ? "" : " ") + std::to_string(e.key);
  return s.empty() ? "(empty)" : s;
}

static Buf* toDevice(const std::vector<KV>& v) {
  Buf* b = pmpp::alloc(v.size() * sizeof(KV));
  if (!v.empty()) pmpp::upload(b, v.data(), v.size() * sizeof(KV));
  return b;
}

// ---------------------------------------------------------------- checks
// co_rank at every k from 0 to m + n, through the provided kernel.
static std::string checkCoRank(const Case& c) {
  const int m = c.m(), n = c.n();
  const std::vector<int> want = coRanks(reference(c)), toB = coRanks(tiesToB(c));
  Buf *A = toDevice(c.A), *B = toDevice(c.B), *out = pmpp::alloc((size_t)(m + n + 1) * 4);
  std::vector<int> got((size_t)(m + n + 1), -1);
  pmpp::upload(out, got.data(), got.size() * 4);
  std::string err = runCoRank(A, m, B, n, out);
  if (err.empty()) err = pmpp::take_error();
  if (!err.empty()) err = "launch error: " + err;
  else {
    pmpp::download(out, got.data(), got.size() * 4);
    for (int k = 0; k <= m + n && err.empty(); ++k) {
      if (got[k] == want[k]) continue;
      char b[200];
      std::snprintf(b, sizeof b, "co_rank(k=%d) returned %d, expected %d", k, got[k], want[k]);
      err = b;
      // The hint only when every value is the ties-to-B co-rank: a starter that returns 0
      // everywhere can match it at a single k by chance.
      if (got == toB) err += " (ties go to A)";
      if (m + n <= 24) err += "; keys A = " + keyList(c.A) + ", B = " + keyList(c.B);
      else err += "; m = " + std::to_string(m) + ", n = " + std::to_string(n);
    }
  }
  pmpp::release(A); pmpp::release(B); pmpp::release(out);
  return err;
}

// C[0 .. total) must equal ref exactly, and C[total ..] must still hold the sentinel.
static std::string checkOutput(const Case& c, const std::vector<KV>& ref, const std::vector<KV>& got, int total) {
  int wrong = 0, first = -1;
  for (int k = 0; k < total; ++k)
    if (!same(got[k], ref[k])) { if (first < 0) first = k; ++wrong; }
  char b[600];
  if (first >= 0) {
    const KV &g = got[first], &r = ref[first];
    const std::string gs = source(g, c), rs = source(r, c);
    if (same(g, kSentinel))
      std::snprintf(b, sizeof b, "C[%d] was never written (it still holds the sentinel)", first);
    else if (g.key == r.key && origin(g, c) == 'B' && origin(r, c) == 'A')
      std::snprintf(b, sizeof b, "C[%d] came from %s; a stable merge takes %s first (both have key %d)", first,
                    gs.c_str(), rs.c_str(), r.key);
    else if (g.key == r.key && origin(g, c) != 0)
      std::snprintf(b, sizeof b, "C[%d] came from %s; a stable merge puts %s there (both have key %d)", first,
                    gs.c_str(), rs.c_str(), r.key);
    else if (origin(g, c) == 0)
      std::snprintf(b, sizeof b, "C[%d] holds %s; expected %s (key %d)", first, gs.c_str(), rs.c_str(), r.key);
    else
      std::snprintf(b, sizeof b, "C[%d] came from %s (key %d); expected %s (key %d)", first, gs.c_str(), g.key,
                    rs.c_str(), r.key);
    return std::string(b) + " (" + std::to_string(wrong) + " of " + std::to_string(total) + " outputs wrong)";
  }
  for (size_t k = (size_t)total; k < got.size(); ++k) {
    if (same(got[k], kSentinel)) continue;
    std::snprintf(b, sizeof b, "C[%zu] was written (now %s), but only the first total = %d outputs may be", k,
                  source(got[k], c).c_str(), total);
    return b;
  }
  return "";
}

// One run of one kernel on device copies of c's inputs. Empty string = correct.
static std::string checkMerge(Kernel kind, const Case& c, Buf* A, Buf* B, const std::vector<KV>& ref, int perThread,
                              int total) {
  const int m = c.m(), n = c.n();
  std::vector<KV> got((size_t)(m + n + kGuard), kSentinel);
  Buf* C = toDevice(got);
  Buf* bounds = pmpp::alloc((size_t)mergeBoundsCount(total, perThread) * 4);
  std::string err = runMerge(kind, A, m, B, n, C, total, perThread, bounds);
  if (err.empty()) {
    err = pmpp::take_error();
    if (!err.empty()) err = "launch error: " + err;
  }
  if (err.empty()) {
    pmpp::download(C, got.data(), got.size() * sizeof(KV));
    err = checkOutput(c, ref, got, total);
  }
  pmpp::release(C); pmpp::release(bounds);
  return err.empty() ? "" : "perThread " + std::to_string(perThread) + ", total " + std::to_string(total) + ": " + err;
}

// total = m + n, (m + n) / 2, 17 and 1, those that are at most m + n.
static std::vector<int> totalsFor(int mn) {
  std::set<int> s;
  for (int t : {mn, mn / 2, 17, 1})
    if (t <= mn) s.insert(t);
  return std::vector<int>(s.rbegin(), s.rend());
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  const bool race = be == "race", emu = be == "emu";
  Rng rng(1301);

  const Case example = makeCase("running example (5, 4)", {1, 7, 8, 9, 10}, {7, 10, 10, 12});

  // (1) co_rank on its own (not on `race`, which runs only the tiled kernel).
  if (!race) {
    t.check("co_rank, A = 1 7 8 9 10, B = 7 10 10 12, k = 0 to 9", [&] { return checkCoRank(example); });
    const int pairs = emu ? 60 : 300;
    Rng r(1302);
    std::vector<Case> random;
    for (int i = 0; i < pairs; ++i) {
      const int m = (int)(r.next() % 41), n = (int)(r.next() % 41);
      std::vector<int> ka = smallKeys((size_t)m, 4, r), kb = smallKeys((size_t)n, 4, r);
      random.push_back(makeCase("", ka, kb));
    }
    t.check("co_rank, " + std::to_string(pairs) + " random pairs (lengths 0 to 40, keys 0 to 3), every k", [&]() -> std::string {
      for (const Case& c : random) {
        std::string err = checkCoRank(c);
        if (!err.empty()) return err;
      }
      return "";
    });
  }

  // (2) Both merges.
  std::vector<Case> cases;
  cases.push_back(example);
  if (!race) {
    cases.push_back(makeCase("(0, 0)", {}, {}));
    cases.push_back(makeCase("(0, 37)", {}, smallKeys(37, 8, rng)));
    cases.push_back(makeCase("(37, 0)", smallKeys(37, 8, rng), {}));
    cases.push_back(makeCase("(1, 1), equal keys", {5}, {5}));
    cases.push_back(makeCase("(100, 100), every key equal", std::vector<int>(100, 42), std::vector<int>(100, 42)));
    cases.push_back(makeCase("(64, 64), A entirely below B", consecutive(0, 64), consecutive(64, 64)));
    cases.push_back(makeCase("(64, 64), A entirely above B", consecutive(64, 64), consecutive(0, 64)));
    cases.push_back(makeCase("(1000, 7), keys in [0, 8)", smallKeys(1000, 8, rng), smallKeys(7, 8, rng)));
    cases.push_back(makeCase("(7, 1000), keys in [0, 8)", smallKeys(7, 8, rng), smallKeys(1000, 8, rng)));
  }
  cases.push_back(makeCase("(300, 257), keys in [0, 8)", smallKeys(300, 8, rng), smallKeys(257, 8, rng)));
  if (!race) {
    std::vector<int> a, b;
    distinctKeys(300, 257, rng, &a, &b);
    cases.push_back(makeCase("(300, 257), distinct keys", a, b));
    cases.push_back(makeCase("(4093, 3001), keys in [0, 8)", smallKeys(4093, 8, rng), smallKeys(3001, 8, rng)));
    a.clear(); b.clear();
    distinctKeys(4093, 3001, rng, &a, &b);
    cases.push_back(makeCase("(4093, 3001), distinct keys", a, b));
  }
  if (!pmpp::is_emu()) {
    cases.push_back(makeCase("(2^20, 2^20), keys in [0, 8)", smallKeys(1 << 20, 8, rng), smallKeys(1 << 20, 8, rng)));
    cases.push_back(makeCase("(3,000,000, 100,000), random 31-bit keys", wideKeys(3000000, rng), wideKeys(100000, rng)));
  }

  // `race` runs only the tiled kernel, small: 2 blocks of 128 threads at perThread 4 on
  // (300, 257), and 3 at perThread 2.
  const std::vector<int> perList = race ? std::vector<int>{4, 2} : emu ? std::vector<int>{1, 3, 8} : std::vector<int>{1, 3, 8, 16};
  const std::vector<Kernel> kinds = race ? std::vector<Kernel>{Kernel::Tiled} : std::vector<Kernel>{Kernel::Basic, Kernel::Tiled};

  for (Kernel kind : kinds) {
    std::vector<int> pers;
    for (int per : perList) {
      std::string why = limitProblem(kind, per);
      if (why.empty()) pers.push_back(per);
      else std::printf("  skip  %s merge, perThread %d: %s\n", kernelName(kind), per, why.c_str());
    }
    for (const Case& c : cases) {
      t.check(std::string(kernelName(kind)) + " merge, " + c.name, [&]() -> std::string {
        const std::vector<KV> ref = reference(c);
        Buf *A = toDevice(c.A), *B = toDevice(c.B);
        std::string err;
        for (int per : pers) {
          for (int total : totalsFor(c.m() + c.n())) {
            err = checkMerge(kind, c, A, B, ref, per, total);
            if (!err.empty()) break;
          }
          if (!err.empty()) break;
        }
        pmpp::release(A); pmpp::release(B);
        return err;
      });
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
struct BenchData {
  int m, n, total;
  Case c;
  std::vector<KV> ref;
  Buf *A, *B, *C, *bounds;
  BenchData(int each, uint32_t seed) : m(each), n(each), total(2 * each) {
    Rng r(seed);
    c = makeCase("bench", wideKeys((size_t)each, r), wideKeys((size_t)each, r));
    A = toDevice(c.A); B = toDevice(c.B);
    C = pmpp::alloc((size_t)total * sizeof(KV));
    bounds = pmpp::alloc((size_t)mergeBoundsCount(total, 1) * 4);   // perThread 1 needs the most
  }
  BenchData(const BenchData&) = delete;
  ~BenchData() { pmpp::release(A); pmpp::release(B); pmpp::release(C); pmpp::release(bounds); }
};

// Fills C with the sentinel, then times one untimed warm-up and `reps` runs, and checks
// the result. Seconds (best run), or < 0 with the reason in *err.
static double timeMerge(const pmpp::Args& a, Kernel kind, BenchData& d, int perThread, std::string* err) {
  std::vector<KV> fill((size_t)d.total, kSentinel);
  pmpp::upload(d.C, fill.data(), fill.size() * sizeof(KV));
  double best = 1e30;
  for (int r = -1; r < a.reps; ++r) {
    pmpp::Timing tm{0, 0};
    std::string e = runMerge(kind, d.A, d.m, d.B, d.n, d.C, d.total, perThread, d.bounds, &tm);
    if (e.empty()) e = pmpp::take_error();
    if (!e.empty()) { *err = e; return -1; }
    if (r >= 0) best = std::min(best, tm.best());
  }
  if (d.ref.empty()) d.ref = reference(d.c);
  std::vector<KV> got((size_t)d.total);
  pmpp::download(d.C, got.data(), got.size() * sizeof(KV));
  got.resize((size_t)d.total + kGuard, kSentinel);   // checkOutput looks past total too
  std::string bad = checkOutput(d.c, d.ref, got, d.total);
  if (!bad.empty()) { *err = "wrong output, not reported: " + bad; return -1; }
  return best;
}

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  // Device bytes for m = n = each: A and B (8 bytes each), C (16), bounds (small).
  auto fits = [&](size_t each) { return mem <= 0 || 32.0 * (double)each <= mem / 4; };
  const std::string basicWhy = limitProblem(Kernel::Basic, 8);
  if (!basicWhy.empty()) { std::printf("bench: the basic kernel cannot run here: %s\n", basicWhy.c_str()); return 1; }

  // Grow m = n from 2^20 until the basic kernel at perThread 8 takes 20 ms, or each input
  // reaches 2^23 elements (or a quarter of the device's memory would be used).
  int each = 1 << 20;
  std::unique_ptr<BenchData> data;
  for (;;) {
    data.reset();
    data.reset(new BenchData(each, 1313));
    std::string err;
    const double s = timeMerge(a, Kernel::Basic, *data, 8, &err);
    if (s < 0) { std::printf("bench: basic merge at perThread 8 failed: %s\n", err.c_str()); return 1; }
    if (s >= 0.020 || each >= (1 << 23) || !fits((size_t)each * 2)) break;
    each *= 2;
  }
  BenchData& d = *data;
  std::printf("m = n = %d (2 x %.0f MB of keys and values), total = m + n = %d, best of %d runs\n", each,
              each * sizeof(KV) / 1e6, d.total, a.reps);
  std::printf("  %-8s %10s %12s %10s\n", "kernel", "perThread", "time ms", "GB/s");
  int rc = 0;
  auto row = [&](Kernel kind, int per) {
    const std::string why = limitProblem(kind, per);
    if (!why.empty()) { std::printf("  %-8s %10d  skipped: %s\n", kernelName(kind), per, why.c_str()); return; }
    std::string e;
    const double s = timeMerge(a, kind, d, per, &e);
    if (s < 0) { std::printf("  %-8s %10d  failed: %s\n", kernelName(kind), per, e.c_str()); rc = 1; return; }
    const double gbps = 2.0 * d.total * sizeof(KV) / s / 1e9;
    std::printf("  %-8s %10d %12.4f %10.1f\n", kernelName(kind), per, s * 1e3, gbps);
    pmpp::bench_row(a, kernelName(kind)).n("per_thread", per).n("m", d.m).n("n", d.n).n("total", d.total)
        .n("seconds", s).n("gb_per_s", gbps).emit();
  };
  for (int per : {1, 2, 4, 8, 16}) row(Kernel::Basic, per);
  for (int per : {4, 8, 16}) row(Kernel::Tiled, per);
  std::printf("  GB/s = 2 x total x %zu bytes / time: each element read once and written once, the least\n"
              "  traffic any merge can make, not what the kernel actually moved. The tiled time includes\n"
              "  its partition launch. Lab 5.1's bandwidth probe measured what this device's memory moves.\n",
              sizeof(KV));
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
