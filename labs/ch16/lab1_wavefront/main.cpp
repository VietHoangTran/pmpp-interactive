// Lab 16.1 driver: tests and benchmarks the square-tile and sheared-tile Smith-Waterman
// kernels, and the provided baseline (one launch per anti-diagonal), against a CPU
// reference. You do not need to edit this file; your work goes in work/.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "pmpp/harness.hpp"
#include "sw.h"

using pmpp::Buf;

enum class Kernel { Baseline, Square, Hyper };
static const char* label(Kernel k) {
  return k == Kernel::Baseline ? "baseline" : k == Kernel::Square ? "square" : "hyper";
}
static std::string longName(Kernel k) {
  return k == Kernel::Baseline ? "baseline (provided)" : k == Kernel::Square ? "square tiles" : "sheared tiles";
}

// ---- entry points implemented in <variant>/ and ../shared/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeSwSquare(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> H, id<MTLBuffer> a,
                    id<MTLBuffer> b, int Lseq, int W, int pad);
void encodeSwHyper(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> H, id<MTLBuffer> a,
                   id<MTLBuffer> b, int Lseq, int W, int pad);
void encodeSwBaseline(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> H, id<MTLBuffer> a,
                      id<MTLBuffer> b, int Lseq);

// A Metal library compiled at run time cannot include sw.h, so its values are passed
// to the compiler as preprocessor definitions.
static std::vector<std::pair<std::string, std::string>> scoring() {
  auto v = [](int x) { return "(" + std::to_string(x) + ")"; };
  return {{"SW_MATCH", v(SW_MATCH)}, {"SW_MISMATCH", v(SW_MISMATCH)}, {"SW_GAP", v(SW_GAP)}};
}
static id<MTLComputePipelineState> pso(Kernel k) {
  if (k == Kernel::Baseline)
    return pmpp::metal::pipeline(std::string(PMPP_METAL_SHARED_DIR) + "/baseline.metal", "sw_baseline", scoring());
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, k == Kernel::Square ? "sw_square" : "sw_hyper", scoring());
}
static bool built(Kernel k) { return pso(k) != nil; }
// What this device allows one threadgroup of kernel k: threads, and threadgroup memory.
static double maxThreads(Kernel k) {
  id<MTLComputePipelineState> p = pso(k);
  return p ? (double)p.maxTotalThreadsPerThreadgroup : 0;
}
static double maxTileBytes(Kernel k) {
  id<MTLComputePipelineState> p = pso(k);
  return p ? pmpp::device_prop("maxThreadgroupMemoryLength", 0) - (double)p.staticThreadgroupMemoryLength : 0;
}
static double tileBytes(int W, int pad) { return std::ceil(4.0 * W * (W + pad) / 16.0) * 16.0; }   // 16-byte multiple
static const char* kGroup = "threadgroup";
static pmpp::Timing runKernel(Kernel k, Buf* H, Buf* A, Buf* B, int lseq, int W, int pad) {
  id<MTLComputePipelineState> p = pso(k);
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* v) {
    using pmpp::metal::buffer;
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    if (k == Kernel::Baseline) encodeSwBaseline(enc, p, buffer(H), buffer(A), buffer(B), lseq);
    else if (k == Kernel::Square) encodeSwSquare(enc, p, buffer(H), buffer(A), buffer(B), lseq, W, pad);
    else encodeSwHyper(enc, p, buffer(H), buffer(A), buffer(B), lseq, W, pad);
  });
}
#else
void launchSwSquare(int* H, const int* a, const int* b, int Lseq, int W, int pad);
void launchSwHyper(int* H, const int* a, const int* b, int Lseq, int W, int pad);
void launchSwBaseline(int* H, const int* a, const int* b, int Lseq);

static bool built(Kernel) { return true; }
static double maxThreads(Kernel) { return pmpp::device_prop("maxThreadsPerBlock", 0); }
static double maxTileBytes(Kernel) { return pmpp::device_prop("sharedMemPerBlock", 0); }
static double tileBytes(int W, int pad) { return 4.0 * W * (W + pad); }
static const char* kGroup = "block";
static pmpp::Timing runKernel(Kernel k, Buf* H, Buf* A, Buf* B, int lseq, int W, int pad) {
  int* h = (int*)pmpp::raw(H);
  const int* a = (const int*)pmpp::raw(A);
  const int* b = (const int*)pmpp::raw(B);
  return pmpp::run([&](void*) {
    if (k == Kernel::Baseline) launchSwBaseline(h, a, b, lseq);
    else if (k == Kernel::Square) launchSwSquare(h, a, b, lseq, W, pad);
    else launchSwHyper(h, a, b, lseq, W, pad);
  });
}
#endif

// Empty if this device can run kernel k with W threads and W * (W + pad) ints of
// shared (threadgroup) memory per group; otherwise the reason it cannot.
static std::string whyNot(Kernel k, int W, int pad) {
  char b[256];
  if (W > maxThreads(k)) {
    std::snprintf(b, sizeof b, "needs %d threads per %s; this device allows %.0f", W, kGroup, maxThreads(k));
    return b;
  }
  if (tileBytes(W, pad) > maxTileBytes(k)) {
    std::snprintf(b, sizeof b, "needs %.0f bytes of tile memory per %s; this device allows %.0f", tileBytes(W, pad),
                  kGroup, maxTileBytes(k));
    return b;
  }
  return "";
}

static int launches(Kernel k, int lseq, int W) {
  if (k == Kernel::Baseline) return 2 * lseq - 1;
  const int T = (lseq + W - 1) / W;
  return k == Kernel::Square ? 2 * T - 1 : 3 * T - 1;
}

// ---------------------------------------------------------------- problems
// Token sequences over a small alphabet, the same on every platform.
static std::vector<int> tokens(int n, uint32_t seed, int alphabet) {
  std::vector<int> v(n);
  uint32_t x = seed * 2654435761u + 12345u;
  for (int i = 0; i < n; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    v[i] = (int)((x >> 8) % (uint32_t)alphabet);
  }
  return v;
}

// The recurrence of sw.h, one cell at a time.
static std::vector<int> cpuReference(const std::vector<int>& a, const std::vector<int>& b) {
  const size_t L = a.size() + 1;
  std::vector<int> H(L * L, 0);
  for (size_t r = 1; r < L; ++r)
    for (size_t q = 1; q < L; ++q) {
      const int s = a[r - 1] == b[q - 1] ? SW_MATCH : SW_MISMATCH;
      H[r * L + q] = std::max({0, H[(r - 1) * L + (q - 1)] + s, H[r * L + (q - 1)] + SW_GAP,
                               H[(r - 1) * L + q] + SW_GAP});
    }
  return H;
}

struct Case {
  int lseq;
  std::string input;
  std::vector<int> a, b, ref;
};

// Six inputs per length, or three (random seed 1, identical, first column) for the
// lengths of `short3`: the CPU emulators pay for every barrier of every thread.
static std::vector<Case> casesFor(const std::vector<int>& lengths, const std::vector<int>& short3 = {}) {
  std::vector<Case> out;
  for (int n : lengths) {
    const bool all = std::find(short3.begin(), short3.end(), n) == short3.end();
    auto add = [&](const std::string& name, std::vector<int> a, std::vector<int> b) {
      Case c{n, name, std::move(a), std::move(b), {}};
      c.ref = cpuReference(c.a, c.b);
      out.push_back(std::move(c));
    };
    for (uint32_t seed = 1; seed <= (all ? 3u : 1u); ++seed)
      add("random, seed " + std::to_string(seed), tokens(n, 100 * seed + n, 4), tokens(n, 100 * seed + n + 50, 4));
    const std::vector<int> same = tokens(n, 7000 + n, 4);
    add("identical sequences", same, same);
    // a uses symbols 0 and 1, b uses 2 and 3: nothing matches and every cell is 0.
    std::vector<int> b = tokens(n, 8000 + n, 2);
    for (int& x : b) x += 2;
    if (all) add("no match (all-zero table)", tokens(n, 8100 + n, 2), b);
    // a leans towards symbol 0 and b[0] is a's most frequent symbol, so table column 1
    // has many matches. Column 1 is where the left edge of the table cuts sheared tiles.
    std::vector<int> a = tokens(n, 9000 + n, 8);
    for (int& x : a) x = x < 5 ? 0 : x - 4;
    int count[4] = {0, 0, 0, 0};
    for (int x : a) count[x]++;
    std::vector<int> b2 = tokens(n, 9100 + n, 4);
    b2[0] = (int)(std::max_element(count, count + 4) - count);
    add("first column matches often", a, b2);
  }
  return out;
}

static const int kPoison = 1234567;      // every interior cell before a run: a cell never written shows
static const int kGuardInts = 256;       // ints allocated after the table
static const int kGuardValue = 0x5EED5EED;

// One problem on the device: the table (row 0 and column 0 zero, interior poisoned,
// then a guard region) and the two sequences.
struct Device {
  int lseq, L;
  Buf *H = nullptr, *A = nullptr, *B = nullptr;
  explicit Device(const Case& c) : lseq(c.lseq), L(c.lseq + 1) {
    H = pmpp::alloc(((size_t)L * L + kGuardInts) * sizeof(int));
    A = pmpp::alloc((size_t)lseq * sizeof(int));
    B = pmpp::alloc((size_t)lseq * sizeof(int));
    pmpp::upload(A, c.a.data(), c.a.size() * sizeof(int));
    pmpp::upload(B, c.b.data(), c.b.size() * sizeof(int));
    reset();
  }
  Device(const Device&) = delete;
  ~Device() { pmpp::release(H); pmpp::release(A); pmpp::release(B); }
  void reset() {
    const size_t cells = (size_t)L * L;
    std::vector<int> init(cells + kGuardInts, kPoison);
    for (int i = 0; i < L; ++i) { init[i] = 0; init[(size_t)i * L] = 0; }
    std::fill(init.begin() + cells, init.end(), kGuardValue);
    pmpp::upload(H, init.data(), init.size() * sizeof(int));
  }
  // Empty string when every cell equals the reference and the guard is untouched.
  std::string check(const std::vector<int>& ref) {
    const size_t cells = (size_t)L * L;
    std::vector<int> got(cells + kGuardInts);
    pmpp::download(H, got.data(), got.size() * sizeof(int));
    size_t bad = 0, first = 0, guard = 0;
    for (size_t i = 0; i < cells; ++i)
      if (got[i] != ref[i]) { if (!bad) first = i; ++bad; }
    for (size_t i = cells; i < got.size(); ++i) guard += got[i] != kGuardValue;
    std::string msg;
    char b[256];
    if (bad) {
      std::snprintf(b, sizeof b, "%zu of %zu cells wrong; first at row %zu, col %zu: got %d, want %d%s", bad, cells,
                    first / L, first % L, got[first], ref[first], got[first] == kPoison ? " (never written)" : "");
      msg = b;
    }
    if (guard) {
      std::snprintf(b, sizeof b, "%s%zu of the %d ints after the table were overwritten", msg.empty() ? "" : "; ",
                    guard, kGuardInts);
      msg += b;
    }
    return msg;
  }
};

// Runs kernel k on every case. Empty string if all are right.
static std::string runCases(Kernel k, const std::vector<Case>& cases, int W, int pad) {
  int failed = 0;
  std::string first;
  for (const Case& c : cases) {
    Device dev(c);
    runKernel(k, dev.H, dev.A, dev.B, c.lseq, W, pad);
    const std::string err = pmpp::take_error();
    const std::string r = err.empty() ? dev.check(c.ref) : "launch error: " + err;
    if (!r.empty()) {
      if (!failed) first = "L_seq " + std::to_string(c.lseq) + ", " + c.input + ": " + r;
      ++failed;
    }
  }
  if (!failed) return "";
  return std::to_string(failed) + " of " + std::to_string(cases.size()) + " cases wrong; first: " + first;
}

static std::string join(const std::vector<int>& v) {
  std::string s;
  for (int x : v) s += (s.empty() ? "" : ", ") + std::to_string(x);
  return s;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  std::vector<int> lengths, short3, widths;
  if (be == "race") { lengths = {9, 20}; widths = {4, 8}; }
  else if (be == "emu") { lengths = {1, 2, 7, 8, 9, 31, 33, 50}; short3 = {31, 33, 50}; widths = {3, 4, 5, 8, 16}; }
  else { lengths = {1, 31, 32, 33, 100, 1000, 2049}; widths = {8, 16, 32, 64}; }
  const std::vector<Case> cases = casesFor(lengths, short3);
  std::printf("each check runs %zu cases:\n"
              "  L_seq in {%s}\n"
              "  inputs: random (3 seeds), identical, no match, first column matching often\n",
              cases.size(), join(lengths).c_str());
  if (!short3.empty())
    std::printf("  (L_seq %s: random seed 1, identical and first column only)\n", join(short3).c_str());
  std::printf("  every cell must equal the CPU reference exactly; the %d ints after the table must stay untouched\n"
              "tile widths W in {%s}, each with pad 0 and 1\n",
              kGuardInts, join(widths).c_str());

  for (Kernel k : {Kernel::Baseline, Kernel::Square, Kernel::Hyper}) {
    if (!built(k)) {
      t.check(longName(k) + ": the kernel builds", [] { return std::string("it did not compile; see the message above"); });
      continue;
    }
    if (k == Kernel::Baseline) {
      t.check(longName(k) + ", one launch per anti-diagonal", [&] { return runCases(k, cases, 0, 0); });
      continue;
    }
    for (int W : widths)
      for (int pad : {0, 1}) {
        const std::string name = longName(k) + ", W " + std::to_string(W) + ", pad " + std::to_string(pad);
        const std::string why = whyNot(k, W, pad);
        if (!why.empty()) { std::printf("  skip  %s: %s\n", name.c_str(), why.c_str()); continue; }
        t.check(name, [&] { return runCases(k, cases, W, pad); });
      }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
static int bench(const pmpp::Args& a) {
  auto timeIt = [&](Kernel k, Device& dev, int W, int pad) -> double {
    const double s = pmpp::time_it(a.reps, [&] { return runKernel(k, dev.H, dev.A, dev.B, dev.lseq, W, pad); }).min_s;
    const std::string err = pmpp::take_error();
    if (!err.empty()) { std::printf("  %s, W %d, pad %d failed: %s\n", label(k), W, pad, err.c_str()); return -1; }
    return s;
  };
  auto problem = [](int lseq) { return Case{lseq, "random", tokens(lseq, 11, 4), tokens(lseq, 12, 4), {}}; };

  // Grow L_seq until the square-tile run at W = 32 takes long enough to time, or 8192.
  const bool w32 = built(Kernel::Square) && whyNot(Kernel::Square, 32, 0).empty();
  int lseq = 1024;
  for (;;) {
    Case c = problem(lseq);
    Device dev(c);
    const double s = w32 ? timeIt(Kernel::Square, dev, 32, 0) : timeIt(Kernel::Baseline, dev, 0, 0);
    if (s < 0) return 1;
    if (s >= 0.020 || lseq >= 8192) break;
    lseq *= 2;
  }
  Case c = problem(lseq);
  Device dev(c);
  const double cells = (double)lseq * lseq;
  std::printf("L_seq %d: %.1f M cell updates per table, best of %d runs\n", lseq, cells / 1e6, a.reps);
  std::printf("  %-9s %4s %4s %9s %11s %8s %12s\n", "kernel", "W", "pad", "launches", "ms", "GCUPS", "vs baseline");

  const double base = timeIt(Kernel::Baseline, dev, 0, 0);
  if (base <= 0) return 1;
  auto row = [&](Kernel k, int W, int pad, double s) {
    const int n = launches(k, lseq, W);
    std::printf("  %-9s %4s %4s %9d %11.3f %8.3f %11.2fx\n", label(k), k == Kernel::Baseline ? "-" : std::to_string(W).c_str(),
                k == Kernel::Baseline ? "-" : std::to_string(pad).c_str(), n, s * 1e3, cells / s / 1e9, base / s);
    pmpp::bench_row(a, label(k)).n("lseq", lseq).n("tile", W).n("pad", pad).n("launches", n).n("seconds", s)
        .n("gcups", cells / s / 1e9).n("speedup_vs_baseline", base / s).emit();
  };
  row(Kernel::Baseline, 0, 0, base);

  double at32[2][2] = {{-1, -1}, {-1, -1}};   // [square, hyper][pad] at W = 32
  for (Kernel k : {Kernel::Square, Kernel::Hyper}) {
    if (!built(k)) { std::printf("  %s: the kernel did not build\n", label(k)); continue; }
    for (int W : {8, 16, 32, 64})
      for (int pad : {0, 1}) {
        const std::string why = whyNot(k, W, pad);
        if (!why.empty()) { std::printf("  %-9s %4d %4d  skipped: %s\n", label(k), W, pad, why.c_str()); continue; }
        const double s = timeIt(k, dev, W, pad);
        if (s <= 0) continue;
        row(k, W, pad, s);
        if (W == 32) at32[k == Kernel::Hyper][pad] = s;
      }
  }

  std::printf("measured on this device at W = 32 (compare with what you predicted):\n");
  auto ratio = [](const char* what, double num, double den) {
    if (num > 0 && den > 0) std::printf("  %-48s %6.2f\n", what, num / den);
  };
  ratio("(1) baseline time / square time", base, at32[0][0]);
  ratio("(2) sheared time / square time, no padding", at32[1][0], at32[0][0]);
  ratio("(3) sheared time, one gap per row / without", at32[1][1], at32[1][0]);
  ratio("    square time, one gap per row / without", at32[0][1], at32[0][0]);
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
