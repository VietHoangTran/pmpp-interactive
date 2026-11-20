// Lab 11.5 driver: tests and benchmarks your single-pass scan with single lookback
// (Figs. 11.17 and 11.18) against Fig. 11.1's loop, and times it beside a copy kernel
// (provided, in ../shared-g13/). CUDA family and the CPU emulators only: there is no
// Metal version of this lab (see README.md). You do not need to edit this file; your
// work goes in work/.
#include <cstring>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

enum Kind { SinglePass, Copy, kKinds };
struct KernelInfo { const char* key; const char* label; };
static const KernelInfo kInfo[kKinds] = {
    {"single_pass", "SinglePassScanKernel (Fig. 11.18)"},
    {"copy", "CopyKernel (provided)"},
};
static const int kGuard = 16;   // guard words after the output, partialSums and flags

// ---- entry points implemented in <variant>/kernel.cu and ../shared-g13/global_copy.cu ----
int coarseFactor();
void launchSinglePassScan(const float* in, float* out, float* partialSums, unsigned int* flags,
                          unsigned int* counter, unsigned int n, unsigned int block);
void launchCopy(const float* in, float* out, unsigned int n, unsigned int block);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static unsigned int* u(Buf* b) { return (unsigned int*)pmpp::raw(b); }

static unsigned int blocksFor(unsigned int n, unsigned int block) {
  const unsigned int segment = (unsigned int)coarseFactor() * block;
  return (n + segment - 1) / segment;
}

// The block fits the device, and the shared memory fits what a block may use without
// opting in (sharedMemPerBlock; ch05-facts A3): the segment buffer, threadSums (one float
// per thread of the largest block), the warp totals and two scalars.
static std::string blockProblem(int k, int block) {
  const int most = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (block > most) return "this device runs at most " + std::to_string(most) + " threads per block";
  if (k == Copy) return "";
  const double limit = pmpp::device_prop("sharedMemPerBlock", 48 * 1024);
  const double need = 1024 * 4 + 32 * 4 + 8 + (double)coarseFactor() * block * 4;
  if (need > limit) {
    char b[200];
    std::snprintf(b, sizeof b, "needs %.0f bytes of shared memory; a block may use %.0f without opting in", need, limit);
    return b;
  }
  return "";
}

struct Buffers { Buf *in, *out, *partialSums, *flags, *counter; };
static void runKernel(int k, const Buffers& b, unsigned int n, unsigned int block, pmpp::Timing* tm = nullptr) {
  pmpp::Timing t = pmpp::run([&](void*) {
    if (k == SinglePass)
      launchSinglePassScan(f(b.in), f(b.out), f(b.partialSums), u(b.flags), u(b.counter), n, block);
    else
      launchCopy(f(b.in), f(b.out), n, block);
  });
  if (tm) *tm = t;
}

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

// As in lab 11.4. Integer inputs whose sum of |x| is at most 2^24 have exact float
// prefix sums in any order of addition, so they must match exactly. Anything else is
// compared within 1e-5 times the sum of |x| up to the element, plus 1e-6: each output is
// reached through at most about 16 additions inside its block plus one per earlier block
// on the chain of partial sums, and each of those is rounded once.
static bool exactData(Data d, const std::vector<float>& x) {
  if (d == Uniform) return false;
  double s = 0;
  for (float v : x) s += std::fabs((double)v);
  return s <= 16777216.0;
}
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
static const uint32_t kSentinel = 0x7F7FDEADu;   // guard words: no scan here produces it
template <class T> static bool isSentinel(T v) { uint32_t w; std::memcpy(&w, &v, 4); return w == kSentinel; }
template <class T> static T sentinelOf() { T v; std::memcpy(&v, &kSentinel, 4); return v; }

// Kernel k on n elements of kind `kind` with blocks of `block`, launched twice on the
// same buffers. Before the first launch every flag is set, partialSums and the output
// are NaN, and the counter is one past the number of blocks, as if left over from other
// work. The second launch gets new data and whatever the first one left.
//
// partialSums and flags get as many guard words as there are blocks, and 16 more, all
// non-zero. A launcher that does not zero the counter then makes its blocks claim
// indices past the end, where every flag they wait on is either a guard word or set by
// the block before: the test fails with a message instead of waiting for ever.
// Empty string = both launches right.
static std::string checkOne(int k, unsigned int n, unsigned int block, Data kind) {
  const unsigned int blocks = blocksFor(n, block);
  const unsigned int segment = (unsigned int)coarseFactor() * block;
  const size_t perBlock = 2 * (size_t)blocks + kGuard;     // entries plus guard words
  Buffers b;
  b.in = pmpp::alloc((size_t)n * 4);
  b.out = pmpp::alloc(((size_t)n + kGuard) * 4);
  b.partialSums = pmpp::alloc(perBlock * 4);
  b.flags = pmpp::alloc(perBlock * 4);
  b.counter = pmpp::alloc((1 + (size_t)kGuard) * 4);
  std::vector<float> got((size_t)n + kGuard, sentinelOf<float>()), ps(perBlock, sentinelOf<float>());
  std::vector<unsigned int> fl(perBlock, sentinelOf<unsigned int>()), ctr(1 + kGuard, sentinelOf<unsigned int>());
  std::fill(got.begin(), got.begin() + n, kNaN);
  std::fill(ps.begin(), ps.begin() + blocks, kNaN);
  std::fill(fl.begin(), fl.begin() + blocks, 1u);
  ctr[0] = blocks + 1;
  pmpp::upload(b.out, got.data(), got.size() * 4);
  pmpp::upload(b.partialSums, ps.data(), ps.size() * 4);
  pmpp::upload(b.flags, fl.data(), fl.size() * 4);
  pmpp::upload(b.counter, ctr.data(), ctr.size() * 4);
  std::string err;
  for (int run = 1; run <= 2 && err.empty(); ++run) {
    std::vector<float> x = makeData(kind, n, 2000u + (uint32_t)(n * 7 + block) + 977u * (uint32_t)run);
    pmpp::upload(b.in, x.data(), (size_t)n * 4);
    char what[220];
    std::snprintf(what, sizeof what, "n %u (%u block%s), %s, run %d of 2: ", n, blocks, blocks == 1 ? "" : "s",
                  dataName(kind), run);
    runKernel(k, b, n, block);
    err = pmpp::take_error();
    if (!err.empty()) { err = what + err; break; }
    pmpp::download(b.out, got.data(), got.size() * 4);
    for (size_t j = n; j < got.size() && err.empty(); ++j)
      if (!isSentinel(got[j]))
        err = std::string(what) + "out[" + std::to_string(j) + "], past the last element, was written. "
              "A thread past n must not store.";
    if (err.empty() && k == SinglePass) {
      pmpp::download(b.partialSums, ps.data(), ps.size() * 4);
      pmpp::download(b.flags, fl.data(), fl.size() * 4);
      pmpp::download(b.counter, ctr.data(), ctr.size() * 4);
      const std::string past = "] was written, but this launch has " + std::to_string(blocks) +
                               (blocks == 1 ? " block" : " blocks") +
                               ": a block used an index past the last one. Was the counter zeroed before the launch?";
      for (size_t j = blocks; j < ps.size() && err.empty(); ++j)
        if (!isSentinel(ps[j])) err = std::string(what) + "partialSums[" + std::to_string(j) + past;
      for (size_t j = blocks; j < fl.size() && err.empty(); ++j)
        if (!isSentinel(fl[j])) err = std::string(what) + "flags[" + std::to_string(j) + past;
      for (size_t j = 1; j < ctr.size() && err.empty(); ++j)
        if (!isSentinel(ctr[j])) err = std::string(what) + "the word after the counter was written.";
      if (err.empty() && ctr[0] != blocks)
        err = std::string(what) + "the block counter is " + std::to_string(ctr[0]) + " after the launch; with one "
              "claim per block it would be " + std::to_string(blocks) + ". Each block takes its index from the "
              "counter (Fig. 11.18), not from blockIdx.x, and the counter starts at 0.";
    }
    if (!err.empty()) break;
    long bad = -1;
    double want = 0;
    if (k == Copy) {
      for (size_t i = 0; i < n && bad < 0; ++i)
        if (!(got[i] == x[i])) { bad = (long)i; want = x[i]; }
    } else {
      bad = firstWrong(x, got.data(), exactData(kind, x), &want);
    }
    if (bad >= 0) {
      char m[300];
      std::snprintf(m, sizeof m, "out[%ld] (segment %lu, position %lu of %u) is %.9g, expected %.9g", bad,
                    (unsigned long)(bad / segment), (unsigned long)(bad % segment), segment, got[bad], want);
      err = std::string(what) + m;
      if (std::isnan(got[bad])) err += " (NaN: nothing was written there, or a partial sum was read before it was written)";
    }
  }
  pmpp::release(b.in); pmpp::release(b.out); pmpp::release(b.partialSums);
  pmpp::release(b.flags); pmpp::release(b.counter);
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
  // At block 64 a segment is coarseFactor()*64 elements: n = 1000 and 4097 are 4 and 17
  // blocks, so on `race` at least four blocks run at once and a block can reach its
  // lookback before the block it waits for has published.
  const std::vector<unsigned int> ns = pick({33, 1000, 4097}, {1, 33, 1000, 4097},
                                            {1, 33, 1000, 4097, 100003, 1u << 22});
  const std::vector<unsigned int> blocks = pick({64}, {64, 256}, {64, 256});
  std::printf("COARSE_FACTOR %d: a block of B threads owns a segment of %d*B elements\n", coarseFactor(),
              coarseFactor());
  for (int k = 0; k < kKinds; ++k) {
    for (unsigned int block : blocks) {
      const std::string name = std::string(kInfo[k].label) + ", block " + std::to_string(block);
      std::string why = blockProblem(k, (int)block);
      if (!why.empty()) {
        std::printf("  skip  %s: %s\n", name.c_str(), why.c_str());
        continue;
      }
      t.check(name, [&]() -> std::string {
        for (unsigned int n : ns)
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
// Best of `reps` timed runs after one untimed warm-up. On CUDA and HIP pmpp::run times
// the host around the launcher and a synchronise, so the launcher's two cudaMemset calls
// are inside the time.
static double bestOf(int reps, const std::function<void(pmpp::Timing*)>& once, std::string* err) {
  double best = 1e30;
  for (int r = -1; r < reps; ++r) {
    pmpp::Timing tm{0, 0};
    once(&tm);
    std::string e = pmpp::take_error();
    if (!e.empty()) { *err = e; return -1; }
    if (r >= 0) best = std::min(best, tm.best());
  }
  return best;
}

static int bench(const pmpp::Args& a) {
  const double mem = pmpp::device_prop("totalGlobalMem", 0);
  const double budget = mem > 0 ? mem / 4 : 1e9;   // bytes the input and output may take
  unsigned int n = 1u << 26;
  while (8.0 * n * 1.01 > budget && n > (1u << 16)) n /= 2;
  std::vector<float> x = pmpp::random_floats(n, 21);
  const unsigned int maxBlocks = blocksFor(n, 256);
  Buffers b;
  b.in = pmpp::alloc((size_t)n * 4);
  b.out = pmpp::alloc((size_t)n * 4);
  b.partialSums = pmpp::alloc((size_t)maxBlocks * 4);
  b.flags = pmpp::alloc((size_t)maxBlocks * 4);
  b.counter = pmpp::alloc(4);
  pmpp::upload(b.in, x.data(), (size_t)n * 4);
  int rc = 0;

  // A quarter of a second of untimed copies first, so the first kernel timed does not
  // also pay for the device waking up.
  {
    double busy = 0;
    for (int i = 0; i < 10000 && busy < 0.25; ++i) {
      pmpp::Timing tm{0, 0};
      runKernel(Copy, b, n, 256, &tm);
      if (!pmpp::take_error().empty()) break;
      busy += tm.wall_s;
    }
  }

  std::printf("n = %u floats (%.0f MB in, %.0f MB out), COARSE_FACTOR %d, best of %d runs\n", n, n * 4 / 1e6,
              n * 4 / 1e6, coarseFactor(), a.reps);
  std::printf("  %-6s %-6s %-36s %9s %9s %9s %8s\n", "block", "S", "kernel", "time ms", "Gelem/s", "GB/s", "of copy");
  std::vector<float> got(n);
  for (unsigned int block : {256u, 1024u}) {
    const unsigned int S = (unsigned int)coarseFactor() * block;
    double copy_s = 0;
    for (int k : {Copy, SinglePass}) {
      std::string why = blockProblem(k, (int)block);
      if (!why.empty()) { std::printf("  %-6u %-6u %-36s skipped: %s\n", block, S, kInfo[k].label, why.c_str()); continue; }
      std::string err;
      double s = bestOf(a.reps, [&](pmpp::Timing* tm) { runKernel(k, b, n, block, tm); }, &err);
      if (s <= 0) { std::printf("  %-6u %-6u %-36s failed: %s\n", block, S, kInfo[k].label, err.c_str()); rc = 1; continue; }
      pmpp::download(b.out, got.data(), (size_t)n * 4);
      double want = 0;
      long bad = -1;
      if (k == Copy) {
        if (std::memcmp(got.data(), x.data(), (size_t)n * 4) != 0) bad = 0;
      } else {
        bad = firstWrong(x, got.data(), false, &want);
      }
      if (bad >= 0) {
        std::printf("  %-6u %-6u %-36s wrong result (first at %ld); not reported\n", block, S, kInfo[k].label, bad);
        rc = 1;
        continue;
      }
      if (k == Copy) copy_s = s;
      const double gbps = 8.0 * n / s / 1e9;
      const double share = copy_s > 0 ? copy_s / s : 0.0;
      std::printf("  %-6u %-6u %-36s %9.3f %9.2f %9.1f %7.0f%%\n", block, S, kInfo[k].label, s * 1e3, n / s / 1e9,
                  gbps, share * 100);
      pmpp::bench_row(a, "global_scan").s("kernel", kInfo[k].key).n("block", block).n("segment", S)
          .n("coarse", coarseFactor()).n("n", n).n("seconds", s).n("elements_per_s", n / s)
          .n("book_bytes_per_element", 8).n("gb_per_s", gbps).n("share_of_copy", share).emit();
    }
  }
  std::printf("  GB/s counts 8 bytes per element (one 4-byte read, one 4-byte write): the book's count for\n"
              "  a single pass (p. 276). \"of copy\" is the copy's time as a share of the kernel's.\n"
              "  Lab 11.4's rows (python3 labs/pmpp.py run 11.4 --bench) are the multi-kernel comparison.\n"
              "\nThese are measurements of this machine, at this size.\n");
  pmpp::release(b.in); pmpp::release(b.out); pmpp::release(b.partialSums);
  pmpp::release(b.flags); pmpp::release(b.counter);
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
