// Lab 11.6 driver (milestone M5, part 1): tests your top-p mass, keep mask and count against a
// CPU reference (Fig. 11.1's loop, in double, made exclusive), tests the provided baseline and
// copy, and benchmarks your kernel against both. You do not need to edit this file; your work
// goes in work/.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <functional>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// The kernels this driver runs. The last two are provided (../shared-g12/).
enum Kernel { Mine, Naive, Copy };
static const int kBlock = 256;   // threads per block your launcher uses (BLOCK_DIM, kBlock in launch.mm)

// ---- entry points implemented in <variant>/ and ../shared-g12/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
bool encodeTopPMask(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> probs,
                    id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept, uint32_t rows, uint32_t cols,
                    float p);
void encodeTopPMaskNaive(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> probs,
                         id<MTLBuffer> massBefore, id<MTLBuffer> keep, id<MTLBuffer> kept, uint32_t rows,
                         uint32_t cols, float p);
void encodeTopPCopy(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> in, id<MTLBuffer> outF,
                    id<MTLBuffer> outB, uint32_t n);
using pmpp::metal::buffer;

// PMPP_METAL_SHARED_DIR is the chapter's shared directory; this lab group's provided files are in shared-g12.
static const std::string kProvided = std::string(PMPP_METAL_SHARED_DIR) + "-g12";

static id<MTLComputePipelineState> pipelineFor(Kernel k) {
  switch (k) {
    case Mine: return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "top_p_mask");
    case Naive: return pmpp::metal::pipeline(kProvided + "/topp_naive.metal", "top_p_mask_naive");
    default: return pmpp::metal::pipeline(kProvided + "/topp_copy.metal", "top_p_copy");
  }
}

// Why this pipeline cannot run the kernel as the lab specifies it, or "". The driver's own copy
// of the rule a correct launch.mm applies: 256 threads per threadgroup, in whole SIMD-groups,
// and no more SIMD-groups than a SIMD-group has lanes.
static std::string pipelineProblem(Kernel k) {
  id<MTLComputePipelineState> p = pipelineFor(k);
  if (!p) return "the Metal kernel did not build: " + pmpp::take_error();
  if (k != Mine) return "";
  const NSUInteger w = p.threadExecutionWidth;
  if ((NSUInteger)kBlock > p.maxTotalThreadsPerThreadgroup || w == 0 || kBlock % w != 0 || (NSUInteger)kBlock / w > w) {
    char b[200];
    std::snprintf(b, sizeof b, "this pipeline (SIMD-group width %lu, at most %lu threads) cannot run %d threads "
                  "as whole SIMD-groups with no more groups than lanes", (unsigned long)w,
                  (unsigned long)p.maxTotalThreadsPerThreadgroup, kBlock);
    return b;
  }
  return "";
}

// Enqueues `repeat` launches of kernel k in one timed run. Empty string when it ran.
static std::string launch(Kernel k, Buf* probs, Buf* mass, Buf* keep, Buf* kept, int rows, int cols, float p,
                          int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(k);
  if (!why.empty()) return why;
  id<MTLComputePipelineState> pso = pipelineFor(k);
  bool ok = true;
  pmpp::Timing t = pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    const uint32_t r = (uint32_t)rows, c = (uint32_t)cols;
    for (int i = 0; i < repeat && ok; ++i) {
      if (k == Mine) ok = encodeTopPMask(enc, pso, buffer(probs), buffer(mass), buffer(keep), buffer(kept), r, c, p);
      else if (k == Naive) encodeTopPMaskNaive(enc, pso, buffer(probs), buffer(mass), buffer(keep), buffer(kept), r, c, p);
      else encodeTopPCopy(enc, pso, buffer(probs), buffer(mass), buffer(keep), r * c);
    }
  });
  if (tm) *tm = t;
  if (!ok) return "your encodeTopPMask refused to dispatch, although this pipeline can run 256 threads per threadgroup";
  return "";
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
#else
void launchTopPMask(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols, float p);
void launchTopPMaskNaive(const float* probs, float* massBefore, unsigned char* keep, int* kept, int rows, int cols,
                         float p);
void launchTopPCopy(const float* in, float* outF, unsigned char* outB, int n);

static std::string pipelineProblem(Kernel k) {
  if (k != Mine) return "";
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  return kBlock <= most ? "" : "this device runs at most " + std::to_string(most) + " threads per block; the lab needs 256";
}
static std::string launch(Kernel k, Buf* probs, Buf* mass, Buf* keep, Buf* kept, int rows, int cols, float p,
                          int repeat = 1, pmpp::Timing* tm = nullptr) {
  std::string why = pipelineProblem(k);
  if (!why.empty()) return why;
  const float* in = (const float*)pmpp::raw(probs);
  float* m = (float*)pmpp::raw(mass);
  unsigned char* kp = (unsigned char*)pmpp::raw(keep);
  int* n = (int*)pmpp::raw(kept);
  pmpp::Timing t = pmpp::run([&](void*) {
    for (int i = 0; i < repeat; ++i) {
      if (k == Mine) launchTopPMask(in, m, kp, n, rows, cols, p);
      else if (k == Naive) launchTopPMaskNaive(in, m, kp, n, rows, cols, p);
      else launchTopPCopy(in, m, kp, rows * cols);
    }
  });
  if (tm) *tm = t;
  return "";
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- tolerances (README.md)
// massBefore is compared with |got - ref| <= rtol * ref, where ref is the exact sum (in double)
// of the float probabilities before the token. Every term is >= 0, so each rounding in your
// sum is at most 2^-24 of a partial sum that is at most ref. For the suggested design (blocks of
// 256, 4 per thread, so 126 chunks at 128,256 tokens) a token's mass passes through at most
// 258 roundings, 1.5e-5 relative; kRtol leaves six times that, enough for a design with
// chunks four times smaller. The provided baseline adds a whole row in one thread, one rounding
// per token, so it is held to cols * 2^-24 more.
static const double kRtol = 1e-4;
static double rtolFor(Kernel k, int cols) { return k == Naive ? kRtol + cols * std::ldexp(1.0, -24) : kRtol; }

// ---------------------------------------------------------------- test rows
static const double kTwo24 = 16777216.0;   // dyadic rows are whole multiples of 2^-24

struct Row {
  std::vector<float> probs;
  bool exact;         // every partial sum exact in float, in any order: compared exactly
  std::string what;
};

// Non-increasing whole numbers, each at most `cap`, summing to `sum`; a smooth profile that
// falls by e^decay across the run. Empty when n * cap < sum.
static std::vector<long long> descendingWhole(int n, long long sum, long long cap, double decay) {
  std::vector<long long> k((size_t)n, 0);
  if ((long long)n * cap < sum) return {};
  double tw = 0;
  std::vector<double> w((size_t)n);
  for (int i = 0; i < n; ++i) tw += (w[i] = std::exp(-decay * i / std::max(1, n)));
  long long s = 0;
  for (int i = 0; i < n; ++i) s += (k[i] = std::min(cap, (long long)std::floor(w[i] / tw * (double)sum)));
  // Hand out what the floors left, from the left, never above `cap` or the element before.
  for (long long rem = sum - s; rem > 0;) {
    for (int i = 0; i < n && rem > 0; ++i) {
      const long long add = std::min(rem, (i == 0 ? cap : k[i - 1]) - k[i]);
      if (add > 0) { k[i] += add; rem -= add; }
    }
  }
  return k;
}

// A dyadic row of n tokens summing to exactly 1 whose first m tokens sum to exactly p (p a
// multiple of 2^-24), so that token m-1 is the one that crosses p; the last `zeros` tokens are 0.
// If that is impossible with non-increasing values, m is halved until it is possible.
static std::vector<float> dyadicRow(int n, double p, int m, int zeros) {
  if (n == 1) return {1.0f};
  m = std::max(1, std::min(m, n - 1));
  zeros = std::max(0, std::min(zeros, n - m - 1));
  const long long P = (long long)std::llround(p * kTwo24), Q = (long long)kTwo24 - P;
  for (;; m = std::max(1, m / 2)) {
    std::vector<long long> a = descendingWhole(m, P, P, 1.0);   // flatter, so the rest fits below it
    std::vector<long long> b = descendingWhole(n - m - zeros, Q, a.back(), 2.0);
    if (!b.empty() || m == 1) {
      if (b.empty()) b = descendingWhole(n - m, Q, P, 2.0), zeros = 0;   // m = 1, p small: no zero tail
      std::vector<float> row;
      for (long long v : a) row.push_back((float)(v / kTwo24));
      for (long long v : b) row.push_back((float)(v / kTwo24));
      row.resize((size_t)n, 0.0f);
      return row;
    }
  }
}

// Softmax of logits drawn by `logit`, computed in double over the first `live` tokens (the rest
// exactly 0), rounded to float and sorted in descending order.
static std::vector<float> softmaxRow(int n, int live, uint32_t seed, const std::function<double(int, double)>& logit) {
  std::vector<float> u = pmpp::random_floats((size_t)n, seed);
  std::vector<double> l((size_t)live);
  double mx = -1e300;
  for (int i = 0; i < live; ++i) mx = std::max(mx, l[i] = logit(i, u[i]));
  double s = 0;
  for (int i = 0; i < live; ++i) s += (l[i] = std::exp(l[i] - mx));
  std::vector<float> row((size_t)n, 0.0f);
  for (int i = 0; i < live; ++i) row[i] = (float)(l[i] / s);
  std::sort(row.begin(), row.end(), [](float a, float b) { return a > b; });
  return row;
}

enum Group { Boundary, Softmax, EdgeP };

// The rows a batch is built from, for a row length n. Dyadic rows put the crossing token where
// a kernel is most likely to get it wrong: the first token, inside a lane's run of 4, at the end
// of a chunk of 1024, among tied probabilities, before a zero tail.
static std::vector<Row> recipes(Group g, int n, double p) {
  std::vector<Row> r;
  // The label names the token whose inclusive mass is exactly p (short rows move it).
  auto dy = [&](int m, int zeros, const std::string& what) {
    std::vector<float> v = dyadicRow(n, p, m, zeros);
    double s = 0;
    int at = -1;
    for (int i = 0; i < n && at < 0; ++i)
      if ((s += v[i]) == p) at = i;
    r.push_back({v, true, what + (at >= 0 ? ", inclusive mass = p at token " + std::to_string(at) : "")});
  };
  auto sm = [&](int live, uint32_t seed, const std::function<double(int, double)>& f, const std::string& what) {
    r.push_back({softmaxRow(n, live, seed, f), false, what});
  };
  if (g == Boundary || g == EdgeP) {
    dy(std::max(1, n / 3), 0, "dyadic, crossing a third of the way in");
    dy(1, 0, "dyadic, p equal to the first probability");
    dy(130, 0, "dyadic, crossing inside a lane's run of 4");
    dy(1024, 0, "dyadic, crossing at the end of a chunk of 1024");
    dy(std::max(1, n / 3), n / 3, "dyadic, the last third exactly 0");
    if (n >= 4 && std::fmod(p * 4.0, 1.0) == 0.0) {
      std::vector<float> ties((size_t)n, 0.0f);
      for (int i = 0; i < 4; ++i) ties[i] = 0.25f;
      r.push_back({ties, true, "four tokens of 0.25, then zeros (ties straddle p)"});
    }
  }
  if (g == Softmax || g == EdgeP) {
    const uint32_t s = 977u * (uint32_t)n;
    sm(n, s + 1, [](int, double u) { return 12.0 * u; }, "softmax of logits uniform in [-12, 12]");
    sm(n, s + 2, [](int i, double u) { return 12.0 * u + (i == 0 ? 40.0 : 0.0); }, "peaked: one logit 40 above the rest");
    sm(n, s + 3, [](int, double u) { return u; }, "nearly flat: logits uniform in [-1, 1]");
    sm(std::max(1, (n + 1) / 2), s + 4, [](int, double u) { return 12.0 * u; }, "softmax with the last half exactly 0");
  }
  return r;
}

// The emulators run 256 OS threads per block and switch between them at every barrier and
// shuffle, so they get fewer values of p.
static std::vector<double> pValues(Group g) {
  const bool emu = pmpp::is_emu();
  if (g == Boundary) return {0.5, 0.75};
  if (g == Softmax) return emu ? std::vector<double>{0.5, 0.99} : std::vector<double>{0.1, 0.5, 0.9, 0.95, 0.99};
  return emu ? std::vector<double>{0.0, 1.0} : std::vector<double>{0.0, 1e-7, 1.0, 1.5};
}

// ---------------------------------------------------------------- one launch, checked
static const int kGuard = 64;                  // elements past the end of every output
static const float kGuardF = -12345.5f;
static const unsigned char kNever = 0xAB;      // keep before the launch
static const unsigned char kGuardB = 0x5A;
static const int kGuardI = -77;

static std::string fmt(const char* f, double a = 0, double b = 0, double c = 0) {
  char buf[240];
  std::snprintf(buf, sizeof buf, f, a, b, c);
  return buf;
}

// Runs kernel k (yours or the provided baseline) on rows[first .. first+count) at probability p.
static std::string checkBatch(Kernel k, const std::vector<Row>& all, int first, int count, int cols, double pd) {
  const float p = (float)pd;
  const size_t total = (size_t)count * cols;
  std::vector<float> x(total);
  std::vector<const Row*> row;
  for (int r = 0; r < count; ++r) {
    row.push_back(&all[(size_t)(first + r) % all.size()]);
    std::memcpy(&x[(size_t)r * cols], row.back()->probs.data(), (size_t)cols * sizeof(float));
  }
  Buf* in = pmpp::alloc(total * 4);
  Buf* mass = pmpp::alloc((total + kGuard) * 4);
  Buf* keep = pmpp::alloc(total + kGuard);
  Buf* kept = pmpp::alloc((size_t)(count + kGuard) * 4);
  pmpp::upload(in, x.data(), total * 4);
  std::vector<float> gm(total + kGuard, kGuardF);
  std::fill(gm.begin(), gm.begin() + total, std::numeric_limits<float>::quiet_NaN());
  std::vector<unsigned char> gk(total + kGuard, kGuardB);
  std::fill(gk.begin(), gk.begin() + total, kNever);
  std::vector<int> gn(count + kGuard, kGuardI);
  std::fill(gn.begin(), gn.begin() + count, -1);
  pmpp::upload(mass, gm.data(), gm.size() * 4);
  pmpp::upload(keep, gk.data(), gk.size());
  pmpp::upload(kept, gn.data(), gn.size() * 4);

  std::string err = launch(k, in, mass, keep, kept, count, cols, p);
  if (err.empty()) err = pmpp::take_error();
  if (!err.empty()) err = "launch error: " + err;
  if (err.empty()) {
    pmpp::download(mass, gm.data(), gm.size() * 4);
    pmpp::download(keep, gk.data(), gk.size());
    pmpp::download(kept, gn.data(), gn.size() * 4);
    const double rtol = rtolFor(k, cols);
    for (int r = 0; r < count && err.empty(); ++r) {
      const float* px = &x[(size_t)r * cols];
      const float* m = &gm[(size_t)r * cols];
      const unsigned char* kp = &gk[(size_t)r * cols];
      const std::string at = std::to_string(count) + (count == 1 ? " row of " : " rows of ") + std::to_string(cols) +
                             ", p = " + fmt("%g", pd) + ", row " + std::to_string(r) + " (" + row[r]->what + ")";
      int ones = 0;
      double ref = 0.0;   // Fig. 11.1's loop in double, made exclusive: the exact mass before token i
      for (int i = 0; i < cols && err.empty(); ++i, ref += px[i - 1]) {
        const double band = row[r]->exact ? 0.0 : rtol * ref;
        const std::string tok = at + ", token " + std::to_string(i) + ": ";
        if (std::isnan(m[i])) { err = tok + "massBefore is NaN (never written)"; break; }
        if (i == 0 && m[i] != 0.0f) { err = tok + fmt("massBefore = %.9g; nothing comes before token 0, so it must be 0 exactly", m[i]); break; }
        if (!(std::fabs((double)m[i] - ref) <= band)) {
          err = tok + fmt("massBefore = %.9g, the exact mass before it is %.9g", m[i], ref);
          err += row[r]->exact ? " (this row's sums are exact in float in any order: no rounding excuse)"
                               : fmt(" (off by %.3g relative; the band is %.3g)", std::fabs(m[i] - ref) / ref, rtol);
          if (std::fabs((double)m[i] - (ref + px[i])) <= band) err += ". That is the INCLUSIVE sum: the scan must exclude the token itself";
          break;
        }
        if (kp[i] > 1) {
          err = tok + "keep = " + std::to_string(kp[i]) + (kp[i] == kNever ? " (0xAB: never written)" : " (must be 0 or 1)");
          break;
        }
        const bool contract = i == 0 || p >= 1.0f || m[i] < p;   // the rule, applied to the mass you wrote
        if ((kp[i] == 1) != contract) {
          err = tok + "keep = " + std::to_string(kp[i]) + fmt(", but your massBefore there is %.9g and p is %.9g", m[i], p);
          err += i == 0 ? " (token 0 is always kept)" : p >= 1.0f ? " (p >= 1 keeps every token)"
                                                     : " (keep exactly when massBefore < p)";
          break;
        }
        // Outside the band the right decision is known whatever your rounding: check it.
        if (i > 0 && p < 1.0f && std::fabs(ref - (double)p) > band && (kp[i] == 1) != (ref < (double)p)) {
          err = tok + "keep = " + std::to_string(kp[i]) + fmt(" but the exact mass before it is %.9g against p = %.9g", ref, p);
          break;
        }
        ones += kp[i];
      }
      if (err.empty() && gn[r] != ones) {
        err = at + ": kept = " + std::to_string(gn[r]) + ", but the row's keep mask has " + std::to_string(ones) + " ones";
        if (gn[r] == -1) err += " (kept[row] was never written)";
      }
    }
    for (int g = 0; g < kGuard && err.empty(); ++g) {
      if (gm[total + g] != kGuardF) err = "wrote massBefore[" + std::to_string(total + g) + "], past the end";
      else if (gk[total + g] != kGuardB) err = "wrote keep[" + std::to_string(total + g) + "], past the end";
      else if (gn[count + g] != kGuardI) err = "wrote kept[" + std::to_string(count + g) + "], past the last row";
    }
  }
  pmpp::release(in); pmpp::release(mass); pmpp::release(keep); pmpp::release(kept);
  return err;
}

// Every recipe of the group in every row position, at every p of the group. On the emulators
// a batch of one row runs only the first recipe: the batches of several rows hold them all.
static std::string checkGroup(Kernel k, Group g, int cols, const std::vector<int>& rowCounts) {
  for (double p : pValues(g)) {
    const std::vector<Row> all = recipes(g, cols, g == EdgeP ? 0.5 : p);
    for (int rows : rowCounts)
      for (int start = 0; start < (int)all.size() && !(pmpp::is_emu() && rows == 1 && start > 0); start += rows) {
        std::string err = checkBatch(k, all, start, rows, cols, p);
        if (!err.empty()) return err;
      }
  }
  return "";
}

// The provided copy: outF = in and outB = 1, nothing past the end.
static std::string checkCopy(int rows, int cols) {
  const size_t total = (size_t)rows * cols;
  std::vector<float> x = pmpp::random_floats(total, 4242u + (uint32_t)cols);
  Buf* in = pmpp::alloc(total * 4);
  Buf* f = pmpp::alloc((total + kGuard) * 4);
  Buf* b = pmpp::alloc(total + kGuard);
  Buf* unused = pmpp::alloc(4);
  pmpp::upload(in, x.data(), total * 4);
  std::vector<float> gf(total + kGuard, kGuardF);
  std::vector<unsigned char> gb(total + kGuard, kGuardB);
  std::fill(gb.begin(), gb.begin() + total, kNever);
  pmpp::upload(f, gf.data(), gf.size() * 4);
  pmpp::upload(b, gb.data(), gb.size());
  std::string err = launch(Copy, in, f, b, unused, rows, cols, 0.5f);
  if (err.empty()) {
    pmpp::download(f, gf.data(), gf.size() * 4);
    pmpp::download(b, gb.data(), gb.size());
    for (size_t i = 0; i < total + kGuard && err.empty(); ++i) {
      const bool inside = i < total;
      if (inside ? !(gf[i] == x[i] && gb[i] == 1) : (gf[i] != kGuardF || gb[i] != kGuardB))
        err = "element " + std::to_string(i) + (inside ? " not copied" : ": written past the end");
    }
  }
  pmpp::release(in); pmpp::release(f); pmpp::release(b); pmpp::release(unused);
  return err;
}

// ---------------------------------------------------------------- tests
static std::vector<int> pick(const std::vector<int>& race, const std::vector<int>& emu, const std::vector<int>& dev) {
  const std::string b = pmpp::backend();
  return b == "race" ? race : b == "emu" ? emu : dev;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  // The emulators run every emulated thread on the CPU (race: as OS threads under
  // ThreadSanitizer), so their rows stay short. A chunk of the suggested design is 1024
  // tokens, so the race backend's 2100 is three chunks, the emulator's 4099 five.
  const std::vector<int> rowCounts = pick({2}, {1, 3}, {1, 4, 7});
  const std::vector<int> colCounts = pick({1, 33, 257, 1000, 2100}, {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4099},
                                          {1, 2, 31, 32, 33, 255, 256, 257, 1000, 4096, 32000, 50257, 128256});
  const struct { Group g; const char* name; } groups[] = {
      {Boundary, "dyadic rows, p = 0.5 and 0.75, exact"},
      {Softmax, "softmax rows, p from 0.1 to 0.99, within the band"},
      {EdgeP, "p = 0, 1e-7, 1 and 1.5"},
  };
  const std::string why = pipelineProblem(Mine);
  if (!why.empty()) {
    t.check("your kernel: usable on this device", [&] { return why; });
  } else {
    for (auto& g : groups)
      for (int cols : colCounts)
        t.check(std::string(g.name) + ", cols " + std::to_string(cols),
                [&] { return checkGroup(Mine, g.g, cols, rowCounts); });
  }

  // The provided baseline and copy, so that a broken provided file shows here and not in the benchmark.
  for (int cols : pick({33}, {1, 257, 1000}, {1, 257, 4096, 32000})) {
    t.check("provided one-thread-per-row baseline, cols " + std::to_string(cols), [&] {
      for (auto& g : groups) {
        std::string e = checkGroup(Naive, g.g, cols, rowCounts);
        if (!e.empty()) return std::string(g.name) + ": " + e;
      }
      return std::string();
    });
    t.check("provided copy, cols " + std::to_string(cols), [&] { return checkCopy(rowCounts.back(), cols); });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Time per call: best of `reps` timed runs after one untimed run; each run repeats the call
// `calls` times back to back, with `calls` doubled until one run lasts at least 2 ms, so that
// short calls are not lost in the cost of submitting the work.
static double perCall(int reps, const std::function<std::string(int, pmpp::Timing*)>& run, int* calls, std::string* err) {
  *calls = 1;
  pmpp::Timing tm{0, 0};
  for (;;) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    if (tm.best() >= 0.002 || *calls >= 4096) break;
    *calls *= 2;
  }
  double best = 1e30;
  for (int r = 0; r < reps; ++r) {
    *err = run(*calls, &tm);
    if (err->empty()) *err = pmpp::take_error();
    if (!err->empty()) return -1;
    best = std::min(best, tm.best());
  }
  return best / *calls;
}

static int bench(const pmpp::Args& a) {
  const double mem = deviceMemory();
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes one shape's buffers may take
  const struct { int rows, cols; } shapes[] = {{1, 128256}, {8, 128256}, {64, 128256}, {256, 32000}};
  const float p = 0.9f;
  const double bytesPer = 9.0;   // read 4, write 4 + 1
  int rc = 0;
  const std::string why = pipelineProblem(Mine);
  if (!why.empty()) { std::printf("your kernel: %s\n", why.c_str()); return 1; }

  std::printf("\ntop-p mask at p = %.2f on softmax rows, best of %d timed runs\n", p, a.reps);
  std::printf("  %5s %7s  %-6s %6s %10s %9s %8s %9s\n", "rows", "cols", "kernel", "calls", "ms", "GB/s", "of copy",
              "vs naive");
  for (auto& s : shapes) {
    const size_t total = (size_t)s.rows * s.cols;
    if ((double)total * bytesPer > budget) {
      std::printf("  %5d %7d  skipped: larger than a quarter of the device memory reported\n", s.rows, s.cols);
      continue;
    }
    std::vector<float> x(total);
    for (int r = 0; r < s.rows; ++r) {
      std::vector<float> one = softmaxRow(s.cols, s.cols, 31u + 7u * (uint32_t)r, [](int, double u) { return 12.0 * u; });
      std::memcpy(&x[(size_t)r * s.cols], one.data(), (size_t)s.cols * 4);
    }
    Buf* in = pmpp::alloc(total * 4);
    Buf* mass = pmpp::alloc(total * 4);
    Buf* keep = pmpp::alloc(total);
    Buf* kept = pmpp::alloc((size_t)s.rows * 4);
    pmpp::upload(in, x.data(), total * 4);
    double copyS = 0, naiveS = 0;
    const struct { Kernel k; const char* key; } order[] = {{Copy, "copy"}, {Naive, "naive"}, {Mine, "yours"}};
    for (auto& o : order) {
      int calls = 0;
      std::string err;
      const double sec = perCall(a.reps, [&](int c, pmpp::Timing* tm) {
        return launch(o.k, in, mass, keep, kept, s.rows, s.cols, p, c, tm);
      }, &calls, &err);
      if (sec <= 0) { std::printf("  %5d %7d  %-6s failed: %s\n", s.rows, s.cols, o.key, err.c_str()); rc = 1; continue; }
      // Check what the timed calls produced before reporting their time.
      std::vector<float> gm(total);
      std::vector<unsigned char> gk(total);
      std::vector<int> gn((size_t)s.rows);
      pmpp::download(mass, gm.data(), total * 4);
      pmpp::download(keep, gk.data(), total);
      pmpp::download(kept, gn.data(), gn.size() * 4);
      std::string bad;
      if (o.k == Copy) {
        for (size_t i = 0; i < total && bad.empty(); ++i)
          if (!(gm[i] == x[i] && gk[i] == 1)) bad = "element " + std::to_string(i) + " not copied";
      } else {
        const double rtol = rtolFor(o.k, s.cols);
        for (int r = 0; r < s.rows && bad.empty(); ++r) {
          double ref = 0;
          int ones = 0;
          for (int i = 0; i < s.cols; ++i) {
            const size_t at = (size_t)r * s.cols + i;
            if (!(std::fabs((double)gm[at] - ref) <= rtol * ref) || gk[at] > 1) { bad = "row " + std::to_string(r) + " wrong"; break; }
            ones += gk[at];
            ref += x[at];
          }
          if (bad.empty() && ones != gn[r]) bad = "row " + std::to_string(r) + ": kept does not match the mask";
        }
      }
      if (!bad.empty()) { std::printf("  %5d %7d  %-6s %s; not reported\n", s.rows, s.cols, o.key, bad.c_str()); rc = 1; continue; }
      const double gbps = total * bytesPer / sec / 1e9;
      if (o.k == Copy) copyS = sec;
      if (o.k == Naive) naiveS = sec;
      std::printf("  %5d %7d  %-6s %6d %10.4f %9.2f", s.rows, s.cols, o.key, calls, sec * 1e3, gbps);
      pmpp::Json j = pmpp::bench_row(a, "top_p_mask");
      j.s("kernel", o.key).n("rows", s.rows).n("cols", s.cols).n("seconds", sec).n("gb_per_s", gbps);
      if (o.k != Copy && copyS > 0) { std::printf(" %7.0f%%", 100.0 * copyS / sec); j.n("share_of_copy", copyS / sec); }
      else std::printf(" %8s", "");
      if (o.k == Mine && naiveS > 0) { std::printf(" %8.1fx", naiveS / sec); j.n("speedup_vs_naive", naiveS / sec); }
      std::printf("\n");
      j.emit();
    }
    pmpp::release(in); pmpp::release(mass); pmpp::release(keep); pmpp::release(kept);
  }
  std::printf("\nGB/s counts the minimum traffic: 9 bytes per token (read 4, write 4 + 1). \"of copy\" is the share\n"
              "of the provided copy's bandwidth on the same shape, \"vs naive\" the speed-up over one thread per row.\n"
              "These are measurements of this machine, at these sizes.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
