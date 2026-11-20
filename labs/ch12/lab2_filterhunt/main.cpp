// Lab 12.2 driver: tests three unstable filters that ship with three defects between them.
// Your work goes in work/. You do not need to edit this file.
//
// The checks are lab 12.1's, at smaller sizes. An unstable filter may leave the kept keys in
// any order (book section 12.2, p. 291), so every call is checked for four things:
//   1. *outputSize equals the number of keys the sequential filter keeps;
//   2. output[0, count) is a permutation of the sequential filter's list: sorted, the two
//      are equal (nothing lost, nothing written twice, nothing invented);
//   3. every slot from count to n still holds the sentinel 0xFFFFFFFF that the driver wrote
//      before the call (keys are below 2^24, so the sentinel is never a key);
//   4. a second call, after the driver sets *outputSize to 0 again and changes nothing
//      else, passes 1 to 3 again.
// On a GPU every input goes through that pair of calls 5 times, because a race can spare
// one launch and not the next. On the emulators, once (under each of emu's three orders).
#include "pmpp/harness.hpp"
using pmpp::Buf;

#ifndef BLOCK_DIM
#define BLOCK_DIM 256
#endif

static const unsigned kSentinel = 0xFFFFFFFFu;
static const unsigned kKeyLimit = 1u << 24;   // keys are in [0, 2^24)

enum class Kernel { Atomic, Warp, Private };
static const char* rowName(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "atomic";
    case Kernel::Warp: return "warp";
    case Kernel::Private: return "private";
  }
  return "?";
}
static const char* figure(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "12.2";
    case Kernel::Warp: return "12.3";
    case Kernel::Private: return "12.6";
  }
  return "?";
}

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeFilterAtomic(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                        id<MTLBuffer> output, uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize);
void encodeFilterWarpAggregated(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                                id<MTLBuffer> output, uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize);
void encodeFilterPrivatised(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                            id<MTLBuffer> output, uint32_t n, uint32_t threshold, id<MTLBuffer> outputSize);
using pmpp::metal::buffer;
static const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "filter_atomic";
    case Kernel::Warp: return "filter_warp_aggregated";
    case Kernel::Private: return "filter_privatised";
  }
  return "?";
}
static void launch(Kernel k, Buf* in, Buf* out, unsigned n, unsigned threshold, Buf* size) {
  id<MTLComputePipelineState> p =
      pmpp::metal::pipeline(PMPP_METAL_SOURCE, kernelName(k), {{"BLOCK_DIM", std::to_string(BLOCK_DIM)}});
  if (!p) return;
  pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    switch (k) {
      case Kernel::Atomic: encodeFilterAtomic(enc, p, buffer(in), buffer(out), n, threshold, buffer(size)); break;
      case Kernel::Warp: encodeFilterWarpAggregated(enc, p, buffer(in), buffer(out), n, threshold, buffer(size)); break;
      case Kernel::Private: encodeFilterPrivatised(enc, p, buffer(in), buffer(out), n, threshold, buffer(size)); break;
    }
  });
}
#else
void launchFilterAtomic(const unsigned int* input, unsigned int* output, unsigned int n,
                        unsigned int threshold, unsigned int* outputSize);
void launchFilterWarpAggregated(const unsigned int* input, unsigned int* output, unsigned int n,
                                unsigned int threshold, unsigned int* outputSize);
void launchFilterPrivatised(const unsigned int* input, unsigned int* output, unsigned int n,
                            unsigned int threshold, unsigned int* outputSize);
static const char* kernelName(Kernel k) {
  switch (k) {
    case Kernel::Atomic: return "filterAtomicKernel";
    case Kernel::Warp: return "filterWarpAggregatedKernel";
    case Kernel::Private: return "filterPrivatisedKernel";
  }
  return "?";
}
static void launch(Kernel k, Buf* in, Buf* out, unsigned n, unsigned threshold, Buf* size) {
  pmpp::run([&](void*) {
    const unsigned* i = (const unsigned*)pmpp::raw(in);
    unsigned* o = (unsigned*)pmpp::raw(out);
    unsigned* s = (unsigned*)pmpp::raw(size);
    switch (k) {
      case Kernel::Atomic: launchFilterAtomic(i, o, n, threshold, s); break;
      case Kernel::Warp: launchFilterWarpAggregated(i, o, n, threshold, s); break;
      case Kernel::Private: launchFilterPrivatised(i, o, n, threshold, s); break;
    }
  });
}
#endif

// ---- inputs, deterministic on every platform (the same as lab 12.1's) ----
static uint32_t xorshift(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }

struct Input {
  std::string name;
  std::vector<unsigned> keys;
  unsigned threshold;            // keep = key < threshold
};

// Keys uniform in [0, 2^24); threshold round(f * 2^24) keeps about a fraction f of them.
static Input fractionInput(size_t n, double f, uint32_t seed) {
  std::vector<unsigned> keys(n);
  uint32_t x = seed * 2654435761u + 1u;
  for (unsigned& k : keys) k = xorshift(x) >> 8;
  char b[64];
  std::snprintf(b, sizeof b, "uniform keys, fraction %g kept", f);
  return Input{b, keys, (unsigned)std::llround(f * kKeyLimit)};
}

// Kept positions chosen by a rule: kept keys below 2^23, the others at or above it. The lane
// patterns put kept keys at lane 31 of every 32-lane warp and in the upper half of every
// 64-lane wavefront; the runs of 40 straddle warp and block boundaries.
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

// `rounds` times: fresh sentinel, then two calls of kernel k on one input. "" if all correct.
static std::string runInput(Kernel k, const Input& in, Buf* dIn, Buf* dOut, Buf* dSize, int rounds) {
  const size_t n = in.keys.size();
  std::vector<unsigned> want = filterSequential(in.keys, in.threshold);
  std::sort(want.begin(), want.end());
  const std::vector<unsigned> sentinel(n, kSentinel);
  pmpp::upload(dIn, in.keys.data(), n * 4);
  std::vector<unsigned> out(n);
  int wrong = 0;
  std::string first;
  for (int r = 0; r < rounds; ++r) {
    pmpp::upload(dOut, sentinel.data(), n * 4);
    for (int call = 1; call <= 2; ++call) {
      const unsigned zero = 0;
      pmpp::upload(dSize, &zero, 4);
      launch(k, dIn, dOut, (unsigned)n, in.threshold, dSize);
      const std::string err = pmpp::take_error();
      if (!err.empty()) return "launch error: " + err;
      unsigned count = 0;
      pmpp::download(dSize, &count, 4);
      pmpp::download(dOut, out.data(), n * 4);
      std::string e = checkResult(in, want, count, out);
      if (!e.empty()) {
        if (wrong++ == 0) first = call == 1 ? e : "second call (after *outputSize was set to 0 again): " + e;
        break;
      }
    }
  }
  if (wrong == 0) return "";
  if (rounds == 1) return first;
  return "wrong in " + std::to_string(wrong) + " of " + std::to_string(rounds) + " rounds; the first: " + first;
}

// Every input of size n through kernel k. "" if all are right.
static std::string runCase(Kernel k, size_t n, int rounds) {
  Buf* dIn = pmpp::alloc(n * 4);
  Buf* dOut = pmpp::alloc(n * 4);
  Buf* dSize = pmpp::alloc(4);
  std::vector<std::string> failures;
  int total = 0;
  for (const Input& in : inputsFor(n)) {
    ++total;
    const std::string e = runInput(k, in, dIn, dOut, dSize, rounds);
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

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string b = pmpp::backend();
  const bool emu = pmpp::is_emu();
  const int rounds = emu ? 1 : 5;
  // The race emulator runs every thread of a block as an OS thread: keep it at n <= 1024.
  std::vector<size_t> sizes = {1, 31, 32, 33, 255, 256, 257, 1000};
  if (b != "race") sizes.push_back(4099);
  if (!emu) {
    sizes.push_back(65537);
    sizes.push_back((size_t)1 << 20);
  }
  const std::string each = rounds > 1 ? std::to_string(rounds) + " rounds of 2 calls each" : "2 calls each";
  for (Kernel k : {Kernel::Atomic, Kernel::Warp, Kernel::Private}) {
    std::printf("%s kernel: %s (book Fig. %s)\n", rowName(k), kernelName(k), figure(k));
    for (size_t n : sizes) {
      t.check(std::string(rowName(k)) + ": n = " + std::to_string(n) + ", 6 fractions and 7 patterns, " + each,
              [&]() { return runCase(k, n, rounds); });
    }
  }
  return t.finish();
}

static int bench(const pmpp::Args&) { std::printf("this lab has no benchmark.\n"); return 0; }

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
