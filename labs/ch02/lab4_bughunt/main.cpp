// Lab 2.4 driver: tests scalePrefix, a host stub and kernel that ship with
// defects. Your work goes in work/. You do not need to edit this file, but you
// may want to read the cases: their order is part of the lab.
//
// Each case owns one host array of n floats and asks scalePrefix to multiply
// the first len of them by 3. Afterwards the whole array is examined: the
// prefix must be scaled, and the padding from len to n must come back with
// exactly the bits it had.
#include <limits>

#include "pmpp/harness.hpp"

// ---- the CPU reference ----
static void cpuScalePrefix(float* x, int len, float s) {
  for (int i = 0; i < len; ++i) x[i] = s * x[i];
}

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pmpp/metal.h"
int scalePrefix(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                float* x_h, uint32_t n, uint32_t len, float s, uint32_t threadsPerGroup);

static id<MTLComputePipelineState> pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "scale_prefix"); }
static id<MTLCommandQueue> queue() {
  static id<MTLCommandQueue> q = [pmpp::metal::device() newCommandQueue];
  return q;
}
static std::string setupError() {
  if (pso()) return "";
  const std::string e = pmpp::take_error();
  return e.empty() ? "no pipeline for kernel scale_prefix" : e;
}
static int callScalePrefix(float* x_h, int n, int len, float s, int group) {
  @autoreleasepool {
    return scalePrefix(pmpp::metal::device(), queue(), pso(), x_h, (uint32_t)n, (uint32_t)len, s, (uint32_t)group);
  }
}
// The largest threadgroup this pipeline accepts.
static int maxGroup() { return (int)pso().maxTotalThreadsPerThreadgroup; }
static const char* groupWords() { return "threads per threadgroup"; }
static void idleDevice() {}
#else
int scalePrefix(float* x_h, int n, int len, float s, int threadsPerBlock);

static std::string setupError() { return ""; }
static int callScalePrefix(float* x_h, int n, int len, float s, int group) {
  return scalePrefix(x_h, n, len, s, group);
}
static int maxGroup() { return (int)pmpp::device_prop("maxThreadsPerBlock", 0); }
static const char* groupWords() { return "threads per block"; }
// Waits for the device and clears the runtime's error state, so that one case
// cannot leave an error behind for the next one to trip over.
static void idleDevice() {
  pmpp::run([](void*) {});
  pmpp::take_error();
}

// The emulator is built with AddressSanitizer, whose leak checker (where the
// platform has one) would report the device array that scalePrefix leaves
// allocated when a CHECK makes it return early. The driver switches the leak
// checker off around the one call that is meant to fail.
#if defined(__SANITIZE_ADDRESS__)
#define PMPP_LAB_ASAN 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer)
#define PMPP_LAB_ASAN 1
#endif
#endif
#if defined(PMPP_LAB_ASAN)
extern "C" {
void __lsan_disable(void);
void __lsan_enable(void);
}
static void leaksExpected(bool on) { if (on) __lsan_disable(); else __lsan_enable(); }
#else
static void leaksExpected(bool) {}
#endif
#endif

static const float kScale = 3.0f;

struct Case { int n, len, group; };
static std::string caseName(const Case& c) {
  return "n = " + std::to_string(c.n) + ", len = " + std::to_string(c.len) + ", " + std::to_string(c.group) + " " + groupWords();
}

// Values in [1, 2]: a scaled element is in [3, 6], so scaled, unscaled and
// never-copied elements cannot be mistaken for one another.
static std::vector<float> makeArray(int n, uint32_t seed) {
  std::vector<float> x = pmpp::random_floats((size_t)n, seed);
  for (float& v : x) v = 1.5f + 0.5f * v;
  return x;
}

// Empty if x is x0 with its first len elements scaled and the rest untouched.
static std::string verify(const std::vector<float>& x, const std::vector<float>& x0, int len) {
  std::vector<float> ref = x0;
  cpuScalePrefix(ref.data(), len, kScale);
  const size_t n = x.size();
  size_t badPrefix = 0, badPadding = 0, first = n;
  for (size_t i = 0; i < n; ++i) {
    bool ok;
    if (i < (size_t)len) ok = std::fabs((double)x[i] - (double)ref[i]) <= 1e-4 * (1.0 + std::fabs((double)ref[i]));
    else ok = std::memcmp(&x[i], &ref[i], sizeof(float)) == 0;      // bit for bit
    if (ok) continue;
    if (i < (size_t)len) badPrefix++; else badPadding++;
    if (first == n) first = i;
  }
  if (first == n) return "";
  char m[400];
  int at = std::snprintf(m, sizeof m, "%zu of the %d prefix elements wrong, %zu of the %zu padding elements changed; first at i = %zu: ",
                         badPrefix, len, badPadding, n - (size_t)len, first);
  if (at < 0 || (size_t)at >= sizeof m) return m;
  if (first < (size_t)len)
    std::snprintf(m + at, sizeof m - (size_t)at, "x_h[i] = %g, expected %g (it was %g, and i < len)",
                  (double)x[first], (double)ref[first], (double)x0[first]);
  else
    std::snprintf(m + at, sizeof m - (size_t)at, "x_h[i] = %g, but it was %g before the call and i >= len",
                  (double)x[first], (double)x0[first]);
  return m;
}

static std::string runCase(const Case& c) {
  idleDevice();
  std::vector<float> x = makeArray(c.n, 900u + (uint32_t)c.n + (uint32_t)c.len);
  const std::vector<float> x0 = x;
  const int rc = callScalePrefix(x.data(), c.n, c.len, kScale, c.group);
  if (rc != 0) return "scalePrefix returned " + std::to_string(rc) + "; a call that works must return 0";
  return verify(x, x0, c.len);
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string broken = setupError();
  if (!broken.empty()) {
    t.check("kernel.metal builds into a pipeline", [&] { return broken; });
    return t.finish();
  }
  const int limit = maxGroup();
  auto runAll = [&](const std::vector<Case>& cases) {
    for (const Case& c : cases) {
      if (limit > 0 && c.group > limit) {
        std::printf("  (skipped: %s; this device allows %d)\n", caseName(c).c_str(), limit);
        continue;
      }
      t.check(caseName(c), [&] { return runCase(c); });
    }
  };

  // Group 1: padding behind the prefix, and enough of it that every thread of
  // every whole block (or threadgroup) still has an element of the array.
  std::printf(" padding behind the prefix:\n");
  std::vector<Case> padded = {{512, 100, 32}, {1024, 700, 256}, {600, 257, 256}, {64, 1, 32}};
  if (!pmpp::is_emu()) padded.push_back({100000, 77777, 256});
  runAll(padded);

  // Group 2: no padding at all. The second case has more threads than elements.
  std::printf(" no padding (len = n):\n");
  runAll({{256, 256, 256}, {1000, 1000, 256}});

#if !defined(PMPP_BACKEND_METAL)
  // Group 3 (CUDA family): a block twice as large as the device allows. The
  // launch statement compiles and returns nothing.
  if (limit > 0) {
    const int tooMany = 2 * limit;
    std::printf(" a launch the device cannot run:\n");
    t.check("a rejected launch (" + std::to_string(tooMany) + " threads per block), then a good one", [&]() -> std::string {
      idleDevice();
      const Case good = {1000, 700, limit < 256 ? limit : 256};
      std::vector<float> x = makeArray(good.n, 31u);
      const std::vector<float> x0 = x;
      std::printf("  (the driver now asks for a block the device cannot run; an error message from CHECK is expected)\n");
      std::fflush(stdout);
      leaksExpected(true);
      const int bad = callScalePrefix(x.data(), good.n, good.len, kScale, tooMany);
      leaksExpected(false);
      if (bad == 0)
        return "scalePrefix returned 0 for a launch the device rejected: the caller was told that everything worked";
      if (std::memcmp(x.data(), x0.data(), x.size() * sizeof(float)) != 0)
        return "scalePrefix reported the failure but had already changed x_h";
      // No idleDevice() here: the next call must cope with whatever the failed one left behind.
      const int rc = callScalePrefix(x.data(), good.n, good.len, kScale, good.group);
      if (rc != 0)
        return "the rejected launch returned " + std::to_string(bad) + " as it should, but the valid call right after it returned " +
               std::to_string(rc) + ": the error of the first call was still there";
      return verify(x, x0, good.len);
    });
  }
#endif
  return t.finish();
}

static int bench(const pmpp::Args&) { std::printf("this lab has no benchmark.\n"); return 0; }

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
