// Lab 12.1 driver: tests your three unstable filters, and the provided copy kernel, against
// a sequential filter; on a GPU it also times all four over a sweep of the fraction of keys
// kept. Your work goes in work/. You do not need to edit this file.
//
// An unstable filter may leave the kept keys in any order (book section 12.2, p. 291), so
// every case checks four things instead of comparing arrays:
//   1. *outputSize equals the number of keys the sequential filter keeps;
//   2. output[0, count) is a permutation of the sequential filter's list: sorted, the two
//      are equal (nothing lost, nothing written twice, nothing invented);
//   3. every slot from count to n still holds the sentinel 0xFFFFFFFF that the driver wrote
//      before the call (keys are below 2^24, so the sentinel is never a key);
//   4. a second call, after the driver sets *outputSize to 0 again and changes nothing
//      else, passes 1 to 3 again.
// Keys are whole numbers, so the comparison is exact: the tolerance is zero.
#include "pmpp/harness.hpp"
using pmpp::Buf;

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

static const unsigned kSentinel = 0xFFFFFFFFu;
static const unsigned kKeyLimit = 1u << 24;   // keys are in [0, 2^24)

// Atomic, Warp and Private are yours; Copy is provided.
enum class Kernel { Atomic, Warp, Private, Copy };
static const char* rowName(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "atomic";
    case Kernel::Warp: return "warp";
    case Kernel::Private: return "private";
    case Kernel::Copy: return "copy";
  }
  return "?";
}
static const char* figure(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "12.2";
    case Kernel::Warp: return "12.3";
    case Kernel::Private: return "12.6";
    case Kernel::Copy: return "-";
  }
  return "?";
}

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeFilterAtomic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                        id<MTLBuffer> output, uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize);
void encodeFilterWarpAggregated(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                                id<MTLBuffer> output, uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize);
void encodeFilterPrivatised(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                            id<MTLBuffer> output, uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize);
void encodeCopyKeys(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                    id<MTLBuffer> output, uint32_t n);
using pmpp::metal::buffer;
static const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "filter_atomic";
    case Kernel::Warp: return "filter_warp_aggregated";
    case Kernel::Private: return "filter_privatised";
    case Kernel::Copy: return "copy_keys";
  }
  return "?";
}
// provided/copy_keys.metal, found from this variant's kernel.metal: <lab>/<variant>/kernel.metal.
static std::string providedMetal() {
  std::string s = PMPP_METAL_SOURCE;
  s = s.substr(0, s.rfind('/'));
  s = s.substr(0, s.rfind('/'));
  return s + "/provided/copy_keys.metal";
}
static pmpp::Timing launch(Kernel k, Buf* in, Buf* out, unsigned n, unsigned threshold, Buf* size) {
  id<MTLComputePipelineState> p = pmpp::metal::pipeline(k == Kernel::Copy ? providedMetal() : PMPP_METAL_SOURCE,
                                                       kernelName(k), {{"BLOCK_DIM", std::to_string(BLOCK_DIM)}});
  if (!p) return pmpp::Timing{0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (k) {
      case Kernel::Atomic: encodeFilterAtomic(enc, p, buffer(in), buffer(out), n, threshold, buffer(size)); break;
      case Kernel::Warp: encodeFilterWarpAggregated(enc, p, buffer(in), buffer(out), n, threshold, buffer(size)); break;
      case Kernel::Private: encodeFilterPrivatised(enc, p, buffer(in), buffer(out), n, threshold, buffer(size)); break;
      case Kernel::Copy: encodeCopyKeys(enc, p, buffer(in), buffer(out), n); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* kBlock = "threadgroup";
#else
void launchFilterAtomic(const unsigned int* input, unsigned int* output, unsigned int n,
                        unsigned int threshold, unsigned int* outputSize);
void launchFilterWarpAggregated(const unsigned int* input, unsigned int* output, unsigned int n,
                                unsigned int threshold, unsigned int* outputSize);
void launchFilterPrivatised(const unsigned int* input, unsigned int* output, unsigned int n,
                            unsigned int threshold, unsigned int* outputSize);
void launchCopyKeys(const unsigned int* input, unsigned int* output, unsigned int n);
static const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "filterAtomicKernel";
    case Kernel::Warp: return "filterWarpAggregatedKernel";
    case Kernel::Private: return "filterPrivatisedKernel";
    case Kernel::Copy: return "copyKeysKernel";
  }
  return "?";
}
static pmpp::Timing launch(Kernel k, Buf* in, Buf* out, unsigned n, unsigned threshold, Buf* size) {
  return pmpp::run([&](void*) {
    const unsigned* i = (const unsigned*)pmpp::raw(in);
    unsigned* o = (unsigned*)pmpp::raw(out);
    unsigned* s = size ? (unsigned*)pmpp::raw(size) : nullptr;   // the copy has no counter
    switch (k) {
      case Kernel::Atomic: launchFilterAtomic(i, o, n, threshold, s); break;
      case Kernel::Warp: launchFilterWarpAggregated(i, o, n, threshold, s); break;
      case Kernel::Private: launchFilterPrivatised(i, o, n, threshold, s); break;
      case Kernel::Copy: launchCopyKeys(i, o, n); break;
    }
  });
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* kBlock = "block";
#endif

// ---- inputs, deterministic on every platform ----
static uint32_t xorshift(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }

struct Input {
  std::string name;              // how the failure messages refer to it
  std::vector<unsigned> keys;
  unsigned threshold;            // keep = key < threshold
};

// Keys uniform in [0, 2^24); threshold round(f * 2^24) keeps about a fraction f of them.
static std::vector<unsigned> uniformKeys(size_t n, uint32_t seed) {
  std::vector<unsigned> keys(n);
  uint32_t x = seed * 2654435761u + 1u;
  for (unsigned& k : keys) k = xorshift(x) >> 8;
  return keys;
}
static unsigned thresholdFor(double f) { return (unsigned)std::llround(f * kKeyLimit); }
static Input fractionInput(size_t n, double f, uint32_t seed) {
  char b[64];
  std::snprintf(b, sizeof b, "uniform keys, fraction %g kept", f);
  return Input{b, uniformKeys(n, seed), thresholdFor(f)};
}

// Kept positions chosen by a rule: kept keys are below 2^23, the others at or above it, and
// the threshold is 2^23. The lane patterns are about where kept keys sit inside a warp: lane
// 31 of every 32-lane warp, and the upper half of every 64-lane wavefront, the lanes a
// 32-bit mask cannot see (notes/today/ch12-facts.md B1). The runs of 40 straddle warp and
// block boundaries.
enum class Pattern { All, None, Last, Lane31, Upper32of64, EveryOther, Runs40 };
static const Pattern kPatterns[] = {Pattern::All, Pattern::None, Pattern::Last, Pattern::Lane31,
                                    Pattern::Upper32of64, Pattern::EveryOther, Pattern::Runs40};
static const char* patternName(Pattern p) {
  switch (p) {
    case Pattern::All: return "every key kept";
    case Pattern::None: return "no key kept";
    case Pattern::Last: return "only the last key kept";
    case Pattern::Lane31: return "only key 31 of every 32 kept";
    case Pattern::Upper32of64: return "only keys 32 to 63 of every 64 kept";
    case Pattern::EveryOther: return "every other key kept";
    case Pattern::Runs40: return "runs of 40 kept, 40 dropped";
  }
  return "?";
}
static bool kept(Pattern p, size_t i, size_t n) {
  switch (p) {
    case Pattern::All: return true;
    case Pattern::None: return false;
    case Pattern::Last: return i == n - 1;
    case Pattern::Lane31: return i % 32 == 31;
    case Pattern::Upper32of64: return i % 64 >= 32;
    case Pattern::EveryOther: return i % 2 == 0;
    case Pattern::Runs40: return (i + 13) % 80 < 40;
  }
  return false;
}
static Input patternInput(size_t n, Pattern p, uint32_t seed) {
  const unsigned half = kKeyLimit / 2;
  std::vector<unsigned> keys(n);
  uint32_t x = seed * 2654435761u + 7u;
  for (size_t i = 0; i < n; ++i) keys[i] = (xorshift(x) >> 9) + (kept(p, i, n) ? 0u : half);
  return Input{patternName(p), keys, half};
}

static const double kTestFractions[] = {0, 0.001, 0.03, 0.5, 0.97, 1};
static std::vector<Input> inputsFor(size_t n) {
  std::vector<Input> v;
  uint32_t seed = 1200u + (uint32_t)n;
  for (double f : kTestFractions) v.push_back(fractionInput(n, f, seed++));
  for (Pattern p : kPatterns) v.push_back(patternInput(n, p, seed++));
  return v;
}

// ---- the CPU reference: a sequential filter ----
// It is stable (it keeps input order); the tests sort both lists before comparing them.
static std::vector<unsigned> filterSequential(const std::vector<unsigned>& keys, unsigned threshold) {
  std::vector<unsigned> out;
  for (unsigned k : keys)
    if (k < threshold) out.push_back(k);
  return out;
}

// "" if (count, out) is a correct unstable filter of `in`; `want` is the reference, sorted.
static std::string checkResult(const Input& in, const std::vector<unsigned>& want, unsigned count,
                               const std::vector<unsigned>& out) {
  const size_t n = in.keys.size();
  char b[300];
  if (count != want.size()) {
    std::snprintf(b, sizeof b, "*outputSize is %u; the sequential filter keeps %zu", count, want.size());
    return b;
  }
  for (size_t s = count; s < n; ++s)
    if (out[s] != kSentinel) {
      std::snprintf(b, sizeof b, "slot %zu, at or past the count %u, was written (it holds %u)", s, count, out[s]);
      return b;
    }
  for (size_t s = 0; s < count; ++s) {
    if (out[s] == kSentinel) {
      std::snprintf(b, sizeof b, "slot %zu of [0, %u) was never written (it still holds the sentinel)", s, count);
      return b;
    }
    if (out[s] >= in.threshold) {
      std::snprintf(b, sizeof b, "slot %zu holds %u, which is not below the threshold %u", s, out[s], in.threshold);
      return b;
    }
  }
  std::vector<unsigned> got(out.begin(), out.begin() + count);
  std::sort(got.begin(), got.end());
  auto mm = std::mismatch(got.begin(), got.end(), want.begin());
  if (mm.first != got.end()) {
    std::snprintf(b, sizeof b,
                  "the kept keys are not all there: sorted, position %zu holds %u where %u was expected "
                  "(a key was lost and another written twice)",
                  (size_t)(mm.first - got.begin()), *mm.first, *mm.second);
    return b;
  }
  return "";
}

// Two calls of kernel k on one input. "" if both are correct.
static std::string runInput(Kernel k, const Input& in, Buf* dIn, Buf* dOut, Buf* dSize) {
  const size_t n = in.keys.size();
  std::vector<unsigned> want = filterSequential(in.keys, in.threshold);
  std::sort(want.begin(), want.end());
  const std::vector<unsigned> sentinel(n, kSentinel);
  pmpp::upload(dIn, in.keys.data(), n * 4);
  pmpp::upload(dOut, sentinel.data(), n * 4);
  std::vector<unsigned> out(n);
  for (int call = 1; call <= 2; ++call) {
    const unsigned zero = 0;
    pmpp::upload(dSize, &zero, 4);
    launch(k, dIn, dOut, (unsigned)n, in.threshold, dSize);
    const std::string err = pmpp::take_error();
    if (!err.empty()) return "launch error: " + err;
    unsigned count = 0;
    pmpp::download(dSize, &count, 4);
    pmpp::download(dOut, out.data(), n * 4);
    const std::string e = checkResult(in, want, count, out);
    if (!e.empty()) return call == 1 ? e : "second call (after *outputSize was set to 0 again): " + e;
  }
  return "";
}

// Every input of size n through kernel k. "" if all are right.
static std::string runCase(Kernel k, size_t n) {
  Buf* dIn = pmpp::alloc(n * 4);
  Buf* dOut = pmpp::alloc(n * 4);
  Buf* dSize = pmpp::alloc(4);
  std::vector<std::string> failures;
  int total = 0;
  for (const Input& in : inputsFor(n)) {
    ++total;
    const std::string e = runInput(k, in, dIn, dOut, dSize);
    if (e.empty()) continue;
    failures.push_back(in.name + ": " + e);
    if (e.rfind("launch error", 0) == 0) break;
  }
  pmpp::release(dIn);
  pmpp::release(dOut);
  pmpp::release(dSize);
  if (failures.empty()) return "";
  std::string out = std::to_string(failures.size()) + " of " + std::to_string(total) + " inputs wrong. ";
  for (size_t i = 0; i < failures.size() && i < 2; ++i) out += (i ? " | " : "") + failures[i];
  if (failures.size() > 2) out += " | and " + std::to_string(failures.size() - 2) + " more";
  return out;
}

// The provided copy kernel: output must equal input.
static std::string runCopy(const std::vector<size_t>& sizes) {
  for (size_t n : sizes) {
    const std::vector<unsigned> keys = uniformKeys(n, 900u + (uint32_t)n);
    Buf* dIn = pmpp::alloc(n * 4);
    Buf* dOut = pmpp::alloc(n * 4);
    pmpp::upload(dIn, keys.data(), n * 4);
    launch(Kernel::Copy, dIn, dOut, (unsigned)n, 0, nullptr);
    std::string err = pmpp::take_error();
    std::vector<unsigned> got(n);
    pmpp::download(dOut, got.data(), n * 4);
    pmpp::release(dIn);
    pmpp::release(dOut);
    if (!err.empty()) return "n = " + std::to_string(n) + ": launch error: " + err;
    for (size_t i = 0; i < n; ++i)
      if (got[i] != keys[i]) return "n = " + std::to_string(n) + ": output[" + std::to_string(i) + "] differs from input";
  }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string b = pmpp::backend();
  // The race emulator runs every thread of a block as an OS thread: keep it at n <= 1024.
  std::vector<size_t> sizes = {1, 31, 32, 33, 255, 256, 257, 1000};
  if (b != "race") sizes.push_back(4099);
  if (!pmpp::is_emu()) {
    sizes.push_back(65537);
    sizes.push_back((size_t)1 << 20);
    sizes.push_back(3 * ((size_t)1 << 20) + 17);
  }
  std::string list;
  for (size_t n : sizes) list += (list.empty() ? "" : ", ") + std::to_string(n);

  std::printf("provided copy kernel (the benchmark's bandwidth reference)\n");
  t.check(std::string("provided ") + kernelName(Kernel::Copy) + ": n = " + list, [&]() { return runCopy(sizes); });

  for (Kernel k : {Kernel::Atomic, Kernel::Warp, Kernel::Private}) {
    std::printf("your %s kernel: %s (book Fig. %s)\n", rowName(k), kernelName(k), figure(k));
    for (size_t n : sizes) {
      t.check(std::string(rowName(k)) + ": n = " + std::to_string(n) + ", 6 fractions and 7 patterns, 2 calls each",
              [&]() { return runCase(k, n); });
    }
  }
  return t.finish();
}

// ---- benchmark ----
// "" if (count, out) is a correct filter of the bench keys: `hist` counts every key value,
// so the kept multiset is hist[0, threshold). Exact, and linear in n.
static std::string checkBench(const std::vector<uint32_t>& hist, unsigned threshold, size_t want, unsigned count,
                              const std::vector<unsigned>& out, std::vector<uint32_t>& work) {
  if (count != want) return "*outputSize is " + std::to_string(count) + ", expected " + std::to_string(want);
  work.assign(hist.begin(), hist.begin() + threshold);
  for (size_t s = 0; s < count; ++s) {
    const unsigned v = out[s];
    if (v >= threshold) return "slot " + std::to_string(s) + " holds a value that is not a kept key";
    if (work[v] == 0) return "key " + std::to_string(v) + " appears more often than in the input";
    --work[v];
  }
  return "";
}

static int bench(const pmpp::Args& a) {
  // n grows from 2^22 until the provided copy takes a few milliseconds: at most 2^26 keys, and
  // input plus output at most a quarter of the memory the device reports.
  const double mem = deviceMemory();
  size_t n = (size_t)1 << 22;
  double copyS = 0;
  for (;;) {
    Buf* dIn = pmpp::alloc(n * 4);
    Buf* dOut = pmpp::alloc(n * 4);
    copyS = pmpp::time_it(a.reps, [&] { return launch(Kernel::Copy, dIn, dOut, (unsigned)n, 0, nullptr); }).min_s;
    pmpp::release(dIn);
    pmpp::release(dOut);
    const std::string err = pmpp::take_error();
    if (!err.empty()) { std::printf("the provided copy kernel failed: %s\n", err.c_str()); return 1; }
    if (copyS >= 3e-3 || n >= ((size_t)1 << 26) || (mem > 0 && 16.0 * n > mem / 4)) break;
    n *= 2;
  }

  const std::vector<unsigned> keys = uniformKeys(n, 12);
  std::vector<uint32_t> hist(kKeyLimit, 0);
  for (unsigned k : keys) ++hist[k];
  Buf* dIn = pmpp::alloc(n * 4);
  Buf* dOut = pmpp::alloc(n * 4);
  Buf* dSize = pmpp::alloc(4);
  pmpp::upload(dIn, keys.data(), n * 4);
  std::vector<unsigned> out(n);
  std::vector<uint32_t> work;
  const unsigned zero = 0;

  std::printf("n = %zu keys, uniform in [0, 2^24); %d threads per %s; best of %d runs.\n", n, BLOCK_DIM, kBlock, a.reps);
  std::printf("GB/s counts the least traffic a filter needs: 4n bytes read, 4 per kept key written, 4 for the counter.\n");

  // The provided copy: 8n bytes, what a filter that keeps every key must move at least.
  {
    const pmpp::Stat st = pmpp::time_it(a.reps, [&] { return launch(Kernel::Copy, dIn, dOut, (unsigned)n, 0, nullptr); });
    pmpp::download(dOut, out.data(), n * 4);
    if (out != keys) {
      std::printf("  copy (provided): WRONG RESULT, not reported\n");
    } else {
      std::printf("  %-18s %9.3f ms %8.1f GB/s %8.2f Gkeys/s\n", "copy (provided)", st.min_s * 1e3,
                  8.0 * n / st.min_s / 1e9, n / st.min_s / 1e9);
      pmpp::bench_row(a, "copy").n("fraction", 1).n("n", (double)n).n("seconds", st.min_s)
          .n("gbps", 8.0 * n / st.min_s / 1e9).n("gkeys_per_s", n / st.min_s / 1e9).emit();
    }
  }

  const double fractions[] = {0, 0.001, 0.01, 0.03, 0.1, 0.25, 0.5, 0.75, 1};
  const int nf = (int)(sizeof fractions / sizeof fractions[0]);
  const Kernel ks[3] = {Kernel::Atomic, Kernel::Warp, Kernel::Private};
  std::vector<std::vector<double>> secs(3, std::vector<double>(nf, 0.0));
  bool allRight = true;
  std::printf("  %-9s %22s %22s %22s\n", "kept", "atomic", "warp", "private");
  for (int fi = 0; fi < nf; ++fi) {
    const double f = fractions[fi];
    const unsigned threshold = thresholdFor(f);
    size_t want = 0;
    for (unsigned v = 0; v < threshold; ++v) want += hist[v];
    const double bytes = 4.0 * n + 4.0 * want + 4.0;
    std::string wrong[3];
    for (int ki = 0; ki < 3; ++ki) {
      const Kernel k = ks[ki];
      // *outputSize is zeroed before every run, outside the timed part.
      const pmpp::Stat st = pmpp::time_it(a.reps, [&] {
        pmpp::upload(dSize, &zero, 4);
        return launch(k, dIn, dOut, (unsigned)n, threshold, dSize);
      });
      const std::string err = pmpp::take_error();
      wrong[ki] = err.empty() ? "" : "launch error: " + err;
      if (wrong[ki].empty()) {   // the last timed run's result is checked before it is reported
        unsigned count = 0;
        pmpp::download(dSize, &count, 4);
        pmpp::download(dOut, out.data(), n * 4);
        wrong[ki] = checkBench(hist, threshold, want, count, out, work);
      }
      if (wrong[ki].empty() && st.min_s <= 0) wrong[ki] = "no time measured";
      if (wrong[ki].empty()) secs[ki][fi] = st.min_s;
      else allRight = false;
    }
    std::printf("  %-9g", f);
    for (int ki = 0; ki < 3; ++ki) {
      if (secs[ki][fi] > 0) std::printf(" %9.3f ms %6.1f GB/s", secs[ki][fi] * 1e3, bytes / secs[ki][fi] / 1e9);
      else std::printf(" %22s", "WRONG, not reported");
    }
    std::printf("\n");
    for (int ki = 0; ki < 3; ++ki) {
      if (!wrong[ki].empty()) {
        std::printf("    %s at fraction %g: %s\n", rowName(ks[ki]), f, wrong[ki].c_str());
        continue;
      }
      pmpp::bench_row(a, rowName(ks[ki])).n("fraction", f).n("n", (double)n).n("seconds", secs[ki][fi])
          .n("gbps", bytes / secs[ki][fi] / 1e9).n("gkeys_per_s", n / secs[ki][fi] / 1e9).emit();
    }
  }
  pmpp::release(dIn);
  pmpp::release(dOut);
  pmpp::release(dSize);

  // The two numbers the README asks you to predict.
  int worst = -1;
  double worstRatio = 0;
  for (int fi = 0; fi < nf; ++fi) {
    if (secs[0][fi] <= 0 || secs[2][fi] <= 0) continue;
    const double r = secs[0][fi] / secs[2][fi];
    if (worst < 0 || r > worstRatio) { worst = fi; worstRatio = r; }
  }
  if (worst >= 0)
    std::printf("  plain atomic against privatised: furthest behind at fraction %g, where it took %.2fx as long\n",
                fractions[worst], worstRatio);
  for (int fi = 0; fi < nf; ++fi)
    if (fractions[fi] == 0.5 && secs[0][fi] > 0 && secs[1][fi] > 0)
      std::printf("  at fraction 0.5: plain atomic time / warp-aggregated time = %.2f\n", secs[0][fi] / secs[1][fi]);
  if (!allRight) std::printf("  WARNING: some results were wrong and are not reported; fix the tests first.\n");
  std::printf("These are measurements of this device. Zeroing *outputSize before each run is not timed.\n");
  return allRight ? 0 : 1;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
