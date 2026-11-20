// Lab 2.1 driver: tests and benchmarks the whole vector addition program.
// You do not need to edit this file; your work goes in work/.
//
// The driver plays the part of main() in the book's Fig. 2.4: it owns three
// ordinary host arrays and calls your vecAdd. It never sees a device pointer.
#include <chrono>
#include <limits>

#include "pmpp/harness.hpp"

// The sequential loop of the book's Fig. 2.4 (p. 26). It is the reference the
// tests compare with, and the CPU side of the benchmark.
__attribute__((noinline))
static void cpuVecAdd(const float* A_h, const float* B_h, float* C_h, int n) {
  for (int i = 0; i < n; ++i) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pmpp/metal.h"
int vecAdd(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
           const float* A_h, const float* B_h, float* C_h, uint32_t n);
id<MTLComputePipelineState> makeVecAddPipeline(id<MTLDevice> device, NSString* mslSource, NSError** error);

// What a Metal program sets up once and keeps: the device, a command queue and
// the pipeline state compiled from the text of kernel.metal.
struct MetalSetup {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
  id<MTLComputePipelineState> pipeline = nil;
  std::string error;
};
static MetalSetup& metalSetup() {
  static MetalSetup s;
  static bool done = false;
  if (done) return s;
  done = true;
  s.device = pmpp::metal::device();
  s.queue = [s.device newCommandQueue];
  if (!s.queue) { s.error = "newCommandQueue returned nil"; return s; }
  NSError* err = nil;
  NSString* source = [NSString stringWithContentsOfFile:[NSString stringWithUTF8String:PMPP_METAL_SOURCE]
                                               encoding:NSUTF8StringEncoding error:&err];
  if (!source) { s.error = std::string("cannot read ") + PMPP_METAL_SOURCE; return s; }
  err = nil;
  s.pipeline = makeVecAddPipeline(s.device, source, &err);
  if (!s.pipeline) {
    s.error = err ? std::string("kernel.metal did not compile: ") + err.localizedDescription.UTF8String
                  : std::string("makeVecAddPipeline returned nil with no error: is the kernel still named vec_add?");
    if (s.error.size() > 600) s.error.resize(600);
  }
  return s;
}
static std::string setupError() { return metalSetup().error; }
static int callVecAdd(const float* A_h, const float* B_h, float* C_h, int n, int /*threadsPerBlock*/) {
  MetalSetup& m = metalSetup();
  @autoreleasepool {
    return vecAdd(m.device, m.queue, m.pipeline, A_h, B_h, C_h, (uint32_t)n);
  }
}
// Metal takes its threadgroup size from the pipeline, so there is one configuration.
static std::vector<int> blockSizes() { return {0}; }
static std::string caseName(int n, int /*threadsPerBlock*/) { return "n = " + std::to_string(n); }
static void idleDevice() {}
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static double largestArray() { return pmpp::device_prop("maxBufferLength", 0); }
#else
int vecAdd(const float* A_h, const float* B_h, float* C_h, int n, int threadsPerBlock);

static std::string setupError() { return ""; }
static int callVecAdd(const float* A_h, const float* B_h, float* C_h, int n, int threadsPerBlock) {
  return vecAdd(A_h, B_h, C_h, n, threadsPerBlock);
}
static int maxThreadsPerBlock() { return (int)pmpp::device_prop("maxThreadsPerBlock", 0); }
// The book's two block sizes (32 on p. 37, 256 on p. 38), if the device allows them.
static std::vector<int> blockSizes() {
  std::vector<int> out;
  for (int b : {32, 256}) if (maxThreadsPerBlock() <= 0 || b <= maxThreadsPerBlock()) out.push_back(b);
  return out;
}
static std::string caseName(int n, int threadsPerBlock) {
  return "n = " + std::to_string(n) + ", " + std::to_string(threadsPerBlock) + " threads per block";
}
// Waits for the device and clears the runtime's error state, so that one test
// case cannot leave an error behind for the next one to trip over.
static void idleDevice() {
  pmpp::run([](void*) {});
  pmpp::take_error();
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static double largestArray() { return 0; }
#endif

// The emulator is built with AddressSanitizer, whose leak checker (where the
// platform has one) would report the device arrays that vecAdd leaves allocated
// when a CHECK makes it return early. That leak is part of the lab's contract,
// so the driver switches the leak checker off around the one call that fails
// on purpose.
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

// ---- one problem instance: host arrays, the reference, and the checks ----
struct Problem {
  int n;
  std::vector<float> a, b, c, ref, a0, b0;
  Problem(int n_, uint32_t seed) : n(n_) {
    // Inputs in [0.5, 1.5], so every correct sum is in [1, 3]: an element the
    // kernel never wrote cannot look right, whatever the device memory held.
    a = pmpp::random_floats((size_t)n, seed);
    b = pmpp::random_floats((size_t)n, seed + 1);
    for (float& v : a) v = 1.0f + 0.5f * v;
    for (float& v : b) v = 1.0f + 0.5f * v;
    a0 = a; b0 = b;
    ref.assign((size_t)n, 0.0f);
    cpuVecAdd(a.data(), b.data(), ref.data(), n);
    reset();
  }
  // C_h starts as NaN: an element still NaN afterwards was never copied back.
  void reset() { c.assign((size_t)n, std::numeric_limits<float>::quiet_NaN()); }
  // Empty string = inputs untouched and C_h equal to the CPU loop's result.
  std::string verify() const {
    if (std::memcmp(a.data(), a0.data(), (size_t)n * sizeof(float)) != 0) return "A_h was modified: the inputs are read-only";
    if (std::memcmp(b.data(), b0.data(), (size_t)n * sizeof(float)) != 0) return "B_h was modified: the inputs are read-only";
    const pmpp::Cmp cmp = pmpp::compare(c.data(), ref.data(), nullptr, (size_t)n);
    if (cmp.ok) return "";
    size_t stillNaN = 0;
    for (float v : c) stillNaN += std::isnan(v) ? 1 : 0;
    const size_t i = cmp.first_bad;
    char m[320];
    int len = std::snprintf(m, sizeof m, "%zu of %d elements wrong; first at i = %zu: C_h[i] = %g, expected %g",
                            cmp.bad, n, i, (double)c[i], (double)ref[i]);
    if (stillNaN > 0 && len > 0 && (size_t)len < sizeof m)
      std::snprintf(m + len, sizeof m - (size_t)len, "; %zu elements still hold the NaN the driver put there, so nothing was copied back to them",
                    stillNaN);
    return m;
  }
};

static std::string runCase(int n, int threadsPerBlock, uint32_t seed) {
  idleDevice();
  Problem p(n, seed);
  const int rc = callVecAdd(p.a.data(), p.b.data(), p.c.data(), n, threadsPerBlock);
  if (rc != 0) return "vecAdd returned " + std::to_string(rc) + "; a call that works must return 0";
  return p.verify();
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string broken = setupError();
  if (!broken.empty()) {
    t.check("kernel.metal builds into a pipeline", [&] { return broken; });
    return t.finish();
  }

  // Lengths on both sides of each block size the tests use, and two that are
  // multiples of nothing. Real devices add one large array.
  std::vector<int> lengths = {1, 31, 32, 33, 255, 256, 257, 1000, 4097};
  if (!pmpp::is_emu()) lengths.push_back((1 << 20) + 3);
  for (int n : lengths)
    for (int tpb : blockSizes())
      t.check(caseName(n, tpb), [&] { return runCase(n, tpb, 100u + (uint32_t)n); });

#if !defined(PMPP_BACKEND_METAL)
  const int maxT = maxThreadsPerBlock();
  if (maxT > 256)
    t.check(caseName(1000, maxT) + " (the largest block this device allows)",
            [&] { return runCase(1000, maxT, 7u); });

  // A block twice as large as the device allows. The launch statement compiles
  // and returns nothing, so only code that asks the runtime finds out.
  if (maxT > 0) {
    const int tooMany = 2 * maxT;
    t.check("a launch the device rejects (" + std::to_string(tooMany) + " threads per block), then a good one",
            [&]() -> std::string {
      idleDevice();
      Problem p(1000, 11u);
      std::printf("  (the driver now asks for a block the device cannot run; an error message from your CHECK is expected)\n");
      std::fflush(stdout);
      leaksExpected(true);
      const int bad = callVecAdd(p.a.data(), p.b.data(), p.c.data(), p.n, tooMany);
      leaksExpected(false);
      if (bad == 0)
        return "vecAdd returned 0 for a launch the device rejected: the launch statement has no value, so ask the runtime right after it";
      // No idleDevice() here: the next call must cope with whatever the failed one left behind.
      p.reset();
      const int good = callVecAdd(p.a.data(), p.b.data(), p.c.data(), p.n, 256 <= maxT ? 256 : maxT);
      if (good != 0)
        return "the rejected launch returned " + std::to_string(bad) + " as it should, but the valid call right after it returned " +
               std::to_string(good) + ": the error of the first call was read without being cleared";
      return p.verify();
    });
  }
#endif
  return t.finish();
}

static double seconds(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

static int bench(const pmpp::Args& a) {
  const std::string broken = setupError();
  if (!broken.empty()) { std::printf("bench skipped: %s\n", broken.c_str()); return 1; }
#if defined(PMPP_BACKEND_METAL)
  const int tpb = 0;
  std::printf("end to end, best of %d calls after one untimed warm-up call\n", a.reps);
#else
  const int tpb = blockSizes().empty() ? maxThreadsPerBlock() : blockSizes().back();
  std::printf("end to end, %d threads per block, best of %d calls after one untimed warm-up call\n", tpb, a.reps);
#endif
  std::printf("  %10s %14s %14s %12s\n", "n", "CPU loop", "your vecAdd", "CPU / yours");
  const double mem = deviceMemory(), oneArray = largestArray();
  for (int shift : {16, 20, 24}) {
    const int n = 1 << shift;
    const double bytes = 4.0 * n;
    // Three device arrays. Keep them well inside what the device reports.
    if ((mem > 0 && 3 * bytes > mem / 4) || (oneArray > 0 && bytes > oneArray)) {
      std::printf("  %10d  skipped: three arrays of %.0f MB are too much for what this device reports\n", n, bytes / 1e6);
      break;
    }
    Problem p(n, 500u + (uint32_t)shift);

    // The CPU loop, writing into memory that has been touched already.
    std::vector<float> cpuOut((size_t)n, 0.0f);
    double cpu_s = 1e300;
    cpuVecAdd(p.a.data(), p.b.data(), cpuOut.data(), n);
    for (int r = 0; r < a.reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      cpuVecAdd(p.a.data(), p.b.data(), cpuOut.data(), n);
      cpu_s = std::min(cpu_s, seconds(t0, std::chrono::steady_clock::now()));
    }

    // Your vecAdd, whole call: allocation, copies, launch, wait, copy back, free.
    // The warm-up call pays for runtime start-up and is checked for correctness.
    idleDevice();
    int rc = callVecAdd(p.a.data(), p.b.data(), p.c.data(), n, tpb);
    if (rc != 0) {
      std::printf("  %10d  stopped: vecAdd returned %d (an allocation of %.0f MB may have failed)\n", n, rc, bytes / 1e6);
      break;
    }
    const std::string wrong = p.verify();
    if (!wrong.empty()) { std::printf("  %10d  stopped: wrong result (%s)\n", n, wrong.c_str()); return 1; }
    double offload_s = 1e300;
    for (int r = 0; r < a.reps && rc == 0; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      rc = callVecAdd(p.a.data(), p.b.data(), p.c.data(), n, tpb);
      offload_s = std::min(offload_s, seconds(t0, std::chrono::steady_clock::now()));
    }
    if (rc != 0) { std::printf("  %10d  stopped: vecAdd returned %d\n", n, rc); break; }

    std::printf("  %10d %11.4f ms %11.4f ms %11.3fx\n", n, cpu_s * 1e3, offload_s * 1e3, cpu_s / offload_s);
    pmpp::bench_row(a, "end_to_end").n("n", n).n("cpu_s", cpu_s).n("offload_s", offload_s)
        .n("speedup_vs_cpu", cpu_s / offload_s).emit();
  }
  std::printf("The last column is above 1 where your GPU version beat the loop. Compare it with what you wrote down.\n");
  std::printf("The CPU loop is one thread, compiled with this build's optimisation flags. Lab 2.2 shows where the time went.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
