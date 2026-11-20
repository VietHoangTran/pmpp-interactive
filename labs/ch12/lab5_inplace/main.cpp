// Lab 12.5 driver (optional lab): tests your single-pass stable filter against a CPU stable
// filter, out of place and in place, and benchmarks it on CUDA and HIP devices. There is no
// Metal version (README.md says why). You do not need to edit this file; your work goes in work/.
#include <atomic>
#include <chrono>
#include <mutex>
#include <thread>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// BLOCK_DIM in kernel.cu: 64 on the race-detecting emulator, 256 everywhere else.
#if defined(PMPP_BACKEND_RACE)
static const uint32_t kBlock = 64;
#else
static const uint32_t kBlock = 256;
#endif
static const uint32_t kThreshold = 0x80000000u;  // keep a key when key < kThreshold
static const uint32_t kSentinel = 0xDEADBEEFu;   // fills the output before an out-of-place run
static const uint32_t kGarbage = 0xCDCDCDCDu;    // fills inclusive[] before a run
static const uint32_t kPad = 64;                 // sentinel slots past the end of the output

void launchStableFilterSinglePass(const unsigned int* input, unsigned int* output, unsigned int n,
                                  unsigned int threshold, unsigned int* outputSize, unsigned int* blockCounter,
                                  unsigned int* flags, unsigned int* inclusive, int coarse);
static unsigned int* u(Buf* b) { return (unsigned int*)pmpp::raw(b); }

// ---------------------------------------------------------------- watchdog
// A look-back that waits on a flag nobody sets never returns. Rather than hang, the driver
// reports the case and exits once a case has run longer than kLimitS seconds.
static const double kLimitS = 120.0;
static std::mutex g_wdMutex;
static std::string g_wdCase;
static std::atomic<long long> g_wdDeadlineMs{0};   // 0: no case running
static long long nowMs() {
  return std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static void watchdogArm(const std::string& what) {
  { std::lock_guard<std::mutex> g(g_wdMutex); g_wdCase = what; }
  g_wdDeadlineMs = nowMs() + (long long)(kLimitS * 1000);
}
static void watchdogDisarm() { g_wdDeadlineMs = 0; }
static void watchdogStart() {
  std::thread([] {
    for (;;) {
      std::this_thread::sleep_for(std::chrono::milliseconds(200));
      const long long d = g_wdDeadlineMs.load();
      if (d != 0 && nowMs() > d) {
        std::string what;
        { std::lock_guard<std::mutex> g(g_wdMutex); what = g_wdCase; }
        const std::string why = "the kernel did not finish within " + std::to_string((int)kLimitS) +
                                " s: a block is probably waiting for a flag that is never set";
        std::printf("  FAIL  %s  -> %s\n", what.c_str(), why.c_str());
        pmpp::Json j;
        j.s("kind", "test").s("lab", PMPP_LAB).s("variant", PMPP_VARIANT).s("backend", pmpp::backend())
         .s("case", what).b("pass", false).s("detail", why);
        j.emit();
        std::fflush(stdout);
        std::_Exit(1);
      }
    }
  }).detach();
}

// ---------------------------------------------------------------- test data
// Kept keys are distinct, below the threshold and not in increasing order: key(i) =
// (i * 2654435761) mod 2^31, one-to-one on [0, 2^31). Dropped keys are 2^31 or above.
static uint32_t keptKey(uint32_t i) { return (i * 2654435761u) & 0x7FFFFFFFu; }

enum Pattern { Random, Tile1Only, OddTilesOnly };
static std::vector<uint32_t> makeKeys(uint32_t n, Pattern p, double fraction, uint32_t tile, uint32_t seed) {
  std::vector<uint32_t> keys(n);
  uint32_t x = seed * 2654435761u + 12345u;
  const double cut = fraction * 4294967296.0;
  for (uint32_t i = 0; i < n; ++i) {
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    bool keep;
    switch (p) {
      case Tile1Only: keep = i / tile == 1; break;          // all of block 1's keys land on block 0's slots
      case OddTilesOnly: keep = (i / tile) % 2 == 1; break;  // every odd block writes over the block before it
      default: keep = fraction >= 1.0 || (double)x < cut; break;
    }
    keys[i] = keep ? keptKey(i) : (0x80000000u | (x >> 1));
  }
  return keys;
}
static std::vector<uint32_t> cpuFilter(const std::vector<uint32_t>& keys) {
  std::vector<uint32_t> out;
  for (uint32_t k : keys) if (k < kThreshold) out.push_back(k);
  return out;
}

// ---------------------------------------------------------------- one problem on the device
struct Problem {
  uint32_t n, blocks;
  int coarse;
  Buf *in, *out, *size, *counter, *flags, *inclusive;
  Problem(uint32_t n_, int coarse_) : n(n_), coarse(coarse_) {
    const uint32_t tile = kBlock * (uint32_t)coarse;
    blocks = (n + tile - 1) / tile;
    in = pmpp::alloc((size_t)n * 4);
    out = pmpp::alloc((size_t)(n + kPad) * 4);
    size = pmpp::alloc(4);
    counter = pmpp::alloc(4);
    flags = pmpp::alloc((size_t)blocks * 4);
    inclusive = pmpp::alloc((size_t)blocks * 4);
  }
  ~Problem() { for (Buf* b : {in, out, size, counter, flags, inclusive}) pmpp::release(b); }
  Problem(const Problem&) = delete;
  static void fill(Buf* b, size_t elems, uint32_t v) {
    std::vector<uint32_t> h(elems, v);
    pmpp::upload(b, h.data(), elems * 4);
  }
  // Fresh input, a zeroed counter and zeroed flags (the driver's job, as in the book, p. 280).
  void prepare(const std::vector<uint32_t>& keys, bool inPlace) {
    pmpp::upload(in, keys.data(), (size_t)n * 4);
    if (!inPlace) fill(out, n + kPad, kSentinel);
    fill(size, 1, kSentinel);
    fill(counter, 1, 0);
    fill(flags, blocks, 0);
    fill(inclusive, blocks, kGarbage);
  }
  pmpp::Timing run(bool inPlace) {
    Buf* dst = inPlace ? in : out;
    return pmpp::run([&](void*) {
      launchStableFilterSinglePass(u(in), u(dst), n, kThreshold, u(size), u(counter), u(flags), u(inclusive), coarse);
    });
  }
  // In place, the array past the count is unspecified (as Thrust's remove_if leaves it,
  // ch12-facts A8) and is not checked; out of place, it must still hold the sentinel.
  std::string check(const std::vector<uint32_t>& keys, bool inPlace) {
    const std::vector<uint32_t> ref = cpuFilter(keys);
    uint32_t count = 0;
    pmpp::download(size, &count, 4);
    char b[300];
    if (count != ref.size()) {
      if (count == kSentinel) return "*outputSize was never written";
      std::snprintf(b, sizeof b, "*outputSize is %u, expected %zu", count, ref.size());
      return b;
    }
    const uint32_t len = inPlace ? n : n + kPad;
    std::vector<uint32_t> got(len);
    pmpp::download(inPlace ? in : out, got.data(), (size_t)len * 4);
    for (uint32_t i = 0; i < count; ++i) {
      if (got[i] != ref[i]) {
        uint32_t src = UINT32_MAX, want = UINT32_MAX;
        for (uint32_t j = 0; j < n && (src == UINT32_MAX || want == UINT32_MAX); ++j) {
          if (keys[j] == got[i]) src = j;
          if (keys[j] == ref[i]) want = j;
        }
        if (src != UINT32_MAX)
          std::snprintf(b, sizeof b, "element %u holds the key from input[%u]; it should hold the key from input[%u]",
                        i, src, want);
        else
          std::snprintf(b, sizeof b, "element %u is 0x%08x, which is no input key; it should hold the key from input[%u]",
                        i, got[i], want);
        return b;
      }
    }
    if (!inPlace) {
      for (uint32_t i = count; i < len; ++i) {
        if (got[i] != kSentinel) {
          std::snprintf(b, sizeof b, "output[%u] was written, but only output[0..%u) belongs to the result", i, count);
          return b;
        }
      }
    }
    return "";
  }
};

static std::string fmtN(uint32_t n) {
  if (n == (1u << 20)) return "2^20";
  if (n == 3u * (1u << 20) + 17) return "3 x 2^20 + 17";
  if (n == (1u << 24)) return "2^24";
  return std::to_string(n);
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string be = pmpp::backend();
  const bool race = be == "race", emu = be == "emu";
  // race: blocks of 64 and n <= 2048, so that several blocks run at once (see kernel.cu).
  const std::vector<uint32_t> sizes = race ? std::vector<uint32_t>{1, 33, 257, 1000, 2048}
                                     : emu ? std::vector<uint32_t>{1, 33, 257, 1000, 4099, 65537}
                                           : std::vector<uint32_t>{1, 33, 257, 1000, 4099, 65537, 1u << 20,
                                                                   3u * (1u << 20) + 17};
  for (int coarse : {1, 2, 4}) {
    const uint32_t tile = kBlock * (uint32_t)coarse;
    for (uint32_t n : sizes) {
      for (bool inPlace : {false, true}) {
        // On the emulator, n = 65537 runs at half kept only (each case takes seconds there).
        const bool small = !(emu && n > 4099);
        const bool severalBlocks = n > tile;   // the two block patterns need a second block
        const std::string name = "coarse " + std::to_string(coarse) + ", n = " + fmtN(n) +
                                 (inPlace ? ", in place" : ", out of place") +
                                 (!small ? ", kept fraction 0.5"
                                         : severalBlocks ? ", kept fractions 0, 0.5, 1, block 1 only, odd blocks only"
                                                         : ", kept fractions 0, 0.5, 1");
        t.check(name, [&]() -> std::string {
          struct Case { Pattern p; double f; const char* what; };
          std::vector<Case> cases = {{Random, 0.5, "kept fraction 0.5"}};
          if (small) {
            cases = {{Random, 0.0, "kept fraction 0"},
                     {Random, 0.5, "kept fraction 0.5"},
                     {Random, 1.0, "kept fraction 1"},
                     {Tile1Only, 0.0, "only block 1's keys kept (they land on block 0's slots)"},
                     {OddTilesOnly, 0.0, "only odd blocks' keys kept"}};
          }
          Problem p(n, coarse);
          for (const Case& c : cases) {
            if (c.p != Random && !severalBlocks) continue;   // one block only: nothing lands on another block
            const std::vector<uint32_t> keys = makeKeys(n, c.p, c.f, tile, 17 * n + (uint32_t)coarse);
            p.prepare(keys, inPlace);
            watchdogArm(name + " [" + c.what + "]");
            p.run(inPlace);
            watchdogDisarm();
            std::string err = p.check(keys, inPlace);
            if (!err.empty()) return std::string(c.what) + ": " + err;
          }
          return "";
        });
      }
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark (CUDA and HIP)
static int bench(const pmpp::Args& a) {
  const uint32_t n = 1u << 24;
  std::printf("n = %s keys (%.0f MB), best of %d runs. GB/s counts the minimum traffic, 4n + 4 x kept bytes.\n"
              "Lab 12.3 part B on the same device is the comparison (its 'staged' rows).\n\n",
              fmtN(n).c_str(), n * 4.0 / 1e6, a.reps);
  std::printf("  %-8s %-6s %-12s %10s %8s\n", "fraction", "coarse", "", "ms", "GB/s");
  int rc = 0;
  for (double f : {0.01, 0.5, 1.0}) {
    const std::vector<uint32_t> keys = makeKeys(n, Random, f, 1, 2024);
    size_t kept = 0;
    for (uint32_t k : keys) kept += k < kThreshold;
    const double minBytes = 4.0 * n + 4.0 * kept;
    for (int coarse : {1, 4, 8}) {
      Problem p(n, coarse);
      for (bool inPlace : {false, true}) {
        p.prepare(keys, inPlace);
        watchdogArm("benchmark");
        p.run(inPlace);
        watchdogDisarm();
        std::string why = p.check(keys, inPlace);   // every timed result is checked first
        if (!why.empty()) {
          std::printf("  %-8g %-6d %-12s not timed, the result is wrong: %s\n", f, coarse,
                      inPlace ? "in place" : "out of place", why.c_str());
          rc = 1;
          continue;
        }
        // Each timed run starts from the original keys and zeroed flags; the reset is not timed.
        const double s = pmpp::time_it(a.reps, [&] {
                           p.prepare(keys, inPlace);
                           return p.run(inPlace);
                         }).min_s;
        const std::string err = pmpp::take_error();
        if (!err.empty() || s <= 0) { std::printf("  coarse %d failed: %s\n", coarse, err.c_str()); rc = 1; continue; }
        std::printf("  %-8g %-6d %-12s %10.4f %8.1f\n", f, coarse, inPlace ? "in place" : "out of place", s * 1e3,
                    minBytes / s / 1e9);
        pmpp::bench_row(a, "singlepass").n("fraction", f).n("kept_fraction", (double)kept / n).n("coarse", coarse)
            .b("inplace", inPlace).n("n", n).n("seconds", s).n("gbps", minBytes / s / 1e9).emit();
      }
    }
  }
  std::printf("\nThese are measurements of this device at this moment, not targets.\n");
  return rc;
}

int main(int argc, char** argv) {
  watchdogStart();
  return pmpp::lab_main(argc, argv, tests, bench);
}
