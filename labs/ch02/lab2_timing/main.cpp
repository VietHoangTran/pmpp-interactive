// Lab 2.2 driver: tests and benchmarks the timed round trip.
// You do not need to edit this file; your work goes in work/.
//
// The driver reads its own clock around every call it makes. Your phases are
// checked against that clock for consistency (they must be real times, in
// seconds, that add up to about one call). Whether each synchronise is in the
// right place is not a test: only the benchmark, on a real device, shows it.
#include <chrono>
#include <limits>

#include "pmpp/harness.hpp"

using pmpp::Buf;

// The sequential loop of the book's Fig. 2.4 (p. 26): the reference for the
// tests and the CPU row of the benchmark.
__attribute__((noinline))
static void cpuVecAdd(const float* A_h, const float* B_h, float* C_h, int n) {
  for (int i = 0; i < n; ++i) {
    C_h[i] = A_h[i] + B_h[i];
  }
}

static double seconds(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

// One call of the reader's vecAddTimed as the driver saw it.
struct Call {
  int rc = 0;
  double wall_s = 0;                       // the driver's clock around the whole call
  double phase[5] = {-1, -1, -1, -1, -1};  // what the call wrote; -1 = never written
};

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pmpp/metal.h"
int vecAddTimed(id<MTLDevice> device, id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                const float* A_h, const float* B_h, float* C_h, uint32_t n, double* phase_s);
int vecAddResident(id<MTLCommandQueue> queue, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> A, id<MTLBuffer> B, id<MTLBuffer> C, uint32_t n);

// phase[0..3] are consecutive pieces of the call; phase[4] is the GPU's own
// report and lies inside the wait.
static const int kWallPhases = 4;
static const char* const kPhaseNames[5] = {"create buffers and copy in", "encode and commit", "wait", "copy out",
                                           "GPU time"};

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
  // dispatchThreads over exactly n threads needs a partial last threadgroup.
  // Apple documents those for GPU family Apple4 and later and for the Metal 3
  // family (notes/today/ch05-facts.md, C5); elsewhere the tests would fail for
  // a reason that is not in your code.
  bool nonuniform = [s.device supportsFamily:MTLGPUFamilyApple4];
  if (@available(macOS 13.0, *)) nonuniform = nonuniform || [s.device supportsFamily:MTLGPUFamilyMetal3];
  if (!nonuniform) {
    s.error = "this GPU reports neither family Apple4 nor Metal 3, so dispatchThreads with a partial threadgroup is not documented for it";
    return s;
  }
  s.queue = [s.device newCommandQueue];
  if (!s.queue) { s.error = "newCommandQueue returned nil"; return s; }
  s.pipeline = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "vec_add");
  if (!s.pipeline) {
    s.error = "kernel.metal did not build into a pipeline: " + pmpp::take_error();
    if (s.error.size() > 600) s.error.resize(600);
  }
  return s;
}
static std::string setupError() { return metalSetup().error; }
static void idleDevice() {}
// Metal has one variant of the call: there is no result to read before the wait.
static std::vector<int> syncModes() { return {1}; }
static std::string modeName(int) { return "host arrays in and out"; }

static Call callTimed(const float* A_h, const float* B_h, float* C_h, int n, int /*syncAfterLaunch*/) {
  MetalSetup& m = metalSetup();
  Call c;
  const auto t0 = std::chrono::steady_clock::now();
  // The pool is inside the timed region on purpose: objects the call left to
  // it (the command buffer among them) are released when it drains, and that
  // time belongs to the call.
  @autoreleasepool {
    c.rc = vecAddTimed(m.device, m.queue, m.pipeline, A_h, B_h, C_h, (uint32_t)n, c.phase);
  }
  c.wall_s = seconds(t0, std::chrono::steady_clock::now());
  return c;
}
// The reader's vecAddResident on driver-owned shared buffers. Returns its status.
static int callResident(Buf* A, Buf* B, Buf* C, int n, double* wall_s) {
  MetalSetup& m = metalSetup();
  int rc;
  const auto t0 = std::chrono::steady_clock::now();
  @autoreleasepool {
    rc = vecAddResident(m.queue, m.pipeline, pmpp::metal::buffer(A), pmpp::metal::buffer(B), pmpp::metal::buffer(C),
                        (uint32_t)n);
  }
  if (wall_s) *wall_s = seconds(t0, std::chrono::steady_clock::now());
  return rc;
}
// The same dispatch issued by the driver through the lab runtime, which reads
// GPUStartTime and GPUEndTime from its command buffer. vecAddResident returns
// only a status, so this is where the resident row's GPU time comes from.
static pmpp::Timing driverDispatch(Buf* A, Buf* B, Buf* C, int n) {
  MetalSetup& m = metalSetup();
  return pmpp::run([&](void* vp) {
    auto* pass = (pmpp::metal::Pass*)vp;
    [pass->encoder setComputePipelineState:m.pipeline];
    [pass->encoder setBuffer:pmpp::metal::buffer(A) offset:0 atIndex:0];
    [pass->encoder setBuffer:pmpp::metal::buffer(B) offset:0 atIndex:1];
    [pass->encoder setBuffer:pmpp::metal::buffer(C) offset:0 atIndex:2];
    NSUInteger group = m.pipeline.maxTotalThreadsPerThreadgroup;
    if (group > (NSUInteger)n) group = (NSUInteger)n;
    [pass->encoder dispatchThreads:MTLSizeMake((NSUInteger)n, 1, 1) threadsPerThreadgroup:MTLSizeMake(group, 1, 1)];
  });
}
static const char* residentName() { return "vecAddResident on shared buffers the driver filled"; }
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static double largestArray() { return pmpp::device_prop("maxBufferLength", 0); }
#else
int vecAddTimed(const float* A_h, const float* B_h, float* C_h, int n, int syncAfterLaunch, double* phase_s);
void launchVecAdd(const float* A_d, const float* B_d, float* C_d, int n);

static const int kWallPhases = 5;
static const char* const kPhaseNames[5] = {"allocation", "copies in", "kernel", "copy back", "free"};

static std::string setupError() { return ""; }
// Waits for the device and clears the runtime's error state, so that one test
// case cannot leave work or an error behind for the next one.
static void idleDevice() {
  pmpp::run([](void*) {});
  pmpp::take_error();
}
static std::vector<int> syncModes() { return {1, 0}; }
static std::string modeName(int sync) { return sync ? "synchronise after the launch" : "no synchronise after the launch"; }

static Call callTimed(const float* A_h, const float* B_h, float* C_h, int n, int syncAfterLaunch) {
  Call c;
  const auto t0 = std::chrono::steady_clock::now();
  c.rc = vecAddTimed(A_h, B_h, C_h, n, syncAfterLaunch, c.phase);
  c.wall_s = seconds(t0, std::chrono::steady_clock::now());
  return c;
}
// launchVecAdd on device arrays, between the runtime's own clock reads: it
// waits for the device, starts the clock, launches, asks whether the launch
// was accepted, waits again and stops the clock.
static pmpp::Timing driverResident(Buf* A, Buf* B, Buf* C, int n) {
  return pmpp::run([&](void*) {
    launchVecAdd((const float*)pmpp::raw(A), (const float*)pmpp::raw(B), (float*)pmpp::raw(C), n);
  });
}
static int callResident(Buf* A, Buf* B, Buf* C, int n, double* wall_s) {
  const pmpp::Timing t = driverResident(A, B, C, n);
  if (wall_s) *wall_s = t.wall_s;
  return 0;
}
static const char* residentName() { return "launchVecAdd on device arrays the driver filled"; }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static double largestArray() { return 0; }
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
  std::string compareWith(const float* got) const {
    const pmpp::Cmp cmp = pmpp::compare(got, ref.data(), nullptr, (size_t)n);
    if (cmp.ok) return "";
    const size_t i = cmp.first_bad;
    char m[240];
    std::snprintf(m, sizeof m, "%zu of %d sums wrong; first at i = %zu: got %g, expected %g",
                  cmp.bad, n, i, (double)got[i], (double)ref[i]);
    return m;
  }
  // Empty string = inputs untouched and C_h equal to the CPU loop's result.
  std::string verify() const {
    if (std::memcmp(a.data(), a0.data(), (size_t)n * sizeof(float)) != 0) return "A_h was modified: the inputs are read-only";
    if (std::memcmp(b.data(), b0.data(), (size_t)n * sizeof(float)) != 0) return "B_h was modified: the inputs are read-only";
    return compareWith(c.data());
  }
};

static double phaseSum(const Call& c) {
  double s = 0;
  for (int k = 0; k < kWallPhases; ++k) s += c.phase[k];
  return s;
}

// Are the phases real times that add up to about one call? Empty string = yes.
static std::string phasesConsistent(const Call& c) {
  char m[320];
  for (int k = 0; k < 5; ++k) {
    if (c.phase[k] == -1.0) {
      std::snprintf(m, sizeof m, "phase_s[%d] (%s) still holds the -1 the driver put there: it was never written",
                    k, kPhaseNames[k]);
      return m;
    }
    if (!std::isfinite(c.phase[k]) || c.phase[k] < 0) {
      std::snprintf(m, sizeof m, "phase_s[%d] (%s) is %g: a phase is a finite number of seconds, not below 0",
                    k, kPhaseNames[k], c.phase[k]);
      return m;
    }
  }
  const double sum = phaseSum(c);
  if (sum > 1.05 * c.wall_s + 50e-6) {
    std::snprintf(m, sizeof m, "your phases add up to %.3f ms but the whole call took %.3f ms on the driver's clock: "
                  "are they in seconds, and does each stretch of the call belong to one phase only?",
                  sum * 1e3, c.wall_s * 1e3);
    return m;
  }
  if (sum < 0.5 * c.wall_s) {
    std::snprintf(m, sizeof m, "your phases add up to %.3f ms, less than half of the %.3f ms the whole call took on the "
                  "driver's clock: part of the call is in no phase, or the values are not seconds",
                  sum * 1e3, c.wall_s * 1e3);
    return m;
  }
#if defined(PMPP_BACKEND_METAL)
  if (c.phase[4] == 0.0)
    return "phase_s[4] (GPU time) is exactly 0: GPUStartTime and GPUEndTime stay 0 until the command buffer has completed";
  if (c.phase[4] > 1.05 * c.wall_s + 50e-6) {
    std::snprintf(m, sizeof m, "phase_s[4] (GPU time) is %.3f ms but the whole call took %.3f ms: is it GPUEndTime - GPUStartTime, in seconds?",
                  c.phase[4] * 1e3, c.wall_s * 1e3);
    return m;
  }
#endif
  return "";
}

static std::string timedCase(int n, int sync, uint32_t seed) {
  Problem p(n, seed);
  // The sums must be right on every attempt. The comparison with the driver's
  // clock gets three: an operating-system pause between the driver's clock read
  // and yours makes one honest call look inconsistent, and says nothing about
  // your code.
  std::string timing;
  for (int attempt = 0; attempt < 3; ++attempt) {
    idleDevice();
    p.reset();
    const Call c = callTimed(p.a.data(), p.b.data(), p.c.data(), n, sync);
    if (c.rc != 0) return "vecAddTimed returned " + std::to_string(c.rc) + "; a call that works must return 0";
    const std::string wrong = p.verify();
    if (!wrong.empty()) return wrong;
    timing = phasesConsistent(c);
    if (timing.empty()) return "";
  }
  return timing;
}

static std::string residentCase(int n, uint32_t seed) {
  idleDevice();
  Problem p(n, seed);
  const size_t bytes = (size_t)n * sizeof(float);
  Buf *A = pmpp::alloc(bytes), *B = pmpp::alloc(bytes), *C = pmpp::alloc(bytes);
  pmpp::upload(A, p.a.data(), bytes);
  pmpp::upload(B, p.b.data(), bytes);
  std::string err;
  const int rc = callResident(A, B, C, n, nullptr);
  if (rc != 0) err = "vecAddResident returned " + std::to_string(rc) + "; a call that works must return 0";
  if (err.empty()) {
    std::vector<float> got((size_t)n), ain((size_t)n), bin((size_t)n);
    pmpp::download(C, got.data(), bytes);
    pmpp::download(A, ain.data(), bytes);
    pmpp::download(B, bin.data(), bytes);
    if (std::memcmp(ain.data(), p.a0.data(), bytes) != 0 || std::memcmp(bin.data(), p.b0.data(), bytes) != 0)
      err = "an input array on the device was modified";
    else
      err = p.compareWith(got.data());
  }
  pmpp::release(A); pmpp::release(B); pmpp::release(C);
  return err;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string broken = setupError();
  if (!broken.empty()) {
    t.check("kernel.metal builds into a pipeline", [&] { return broken; });
    return t.finish();
  }
  // One element, one past a block of 256, two lengths that are multiples of
  // nothing. Real devices add one large array.
  std::vector<int> lengths = {1, 257, 1000, 4097};
  if (!pmpp::is_emu()) lengths.push_back((1 << 20) + 3);
  for (int n : lengths)
    for (int sync : syncModes())
      t.check("n = " + std::to_string(n) + ", " + modeName(sync),
              [&] { return timedCase(n, sync, 200u + (uint32_t)n + (uint32_t)sync); });
  for (int n : {1, 256, 257, 1000, 4097})
    t.check(std::string(residentName()) + ", n = " + std::to_string(n), [&] { return residentCase(n, 300u + (uint32_t)n); });
  return t.finish();
}

// ---- benchmark ----
// The call with the shortest whole-call time out of `reps`, after one untimed
// warm-up call that is checked for correctness. rc != 0 or `wrong` set = stop.
static Call bestCall(Problem& p, int sync, int reps, std::string* wrong) {
  idleDevice();
  Call best = callTimed(p.a.data(), p.b.data(), p.c.data(), p.n, sync);
  if (best.rc != 0) return best;
  *wrong = p.verify();
  if (wrong->empty()) *wrong = phasesConsistent(best);
  if (!wrong->empty()) return best;
  best.wall_s = 1e300;
  for (int r = 0; r < reps; ++r) {
    idleDevice();
    const Call c = callTimed(p.a.data(), p.b.data(), p.c.data(), p.n, sync);
    if (c.rc != 0) return c;
    if (c.wall_s < best.wall_s) best = c;
  }
  return best;
}

static double ms(double s) { return s * 1e3; }

static int bench(const pmpp::Args& a) {
  const std::string broken = setupError();
  if (!broken.empty()) { std::printf("bench skipped: %s\n", broken.c_str()); return 1; }
  std::printf("Each row is one call: the one with the shortest whole-call time out of %d, after one untimed warm-up call.\n", a.reps);
  std::printf("'whole call' is the driver's clock around the call. Everything else in a row is what your code reported.\n");
  const double mem = deviceMemory(), oneArray = largestArray();
  for (int shift : {16, 20, 24}) {
    const int n = 1 << shift;
    const double bytes = 4.0 * n;
    // Three device arrays. Keep them well inside what the device reports.
    if ((mem > 0 && 3 * bytes > mem / 4) || (oneArray > 0 && bytes > oneArray)) {
      std::printf("\nn = %d  skipped: three arrays of %.0f MB are too much for what this device reports\n", n, bytes / 1e6);
      break;
    }
    Problem p(n, 600u + (uint32_t)shift);

    // The CPU loop, writing into memory that has been touched already.
    std::vector<float> cpuOut((size_t)n, 0.0f);
    double cpu_s = 1e300;
    cpuVecAdd(p.a.data(), p.b.data(), cpuOut.data(), n);
    for (int r = 0; r < a.reps; ++r) {
      const auto t0 = std::chrono::steady_clock::now();
      cpuVecAdd(p.a.data(), p.b.data(), cpuOut.data(), n);
      cpu_s = std::min(cpu_s, seconds(t0, std::chrono::steady_clock::now()));
    }

    // The resident path: the data is on the device before the clock starts.
    const size_t sz = (size_t)n * sizeof(float);
    Buf *A = pmpp::alloc(sz), *B = pmpp::alloc(sz), *C = pmpp::alloc(sz);
    if (!pmpp::take_error().empty()) {
      std::printf("\nn = %d  stopped: the driver could not allocate three device arrays of %.0f MB\n", n, bytes / 1e6);
      pmpp::release(A); pmpp::release(B); pmpp::release(C);
      break;
    }
    pmpp::upload(A, p.a.data(), sz);
    pmpp::upload(B, p.b.data(), sz);
    double resident_s = 1e300, w = 0;
    int rrc = callResident(A, B, C, n, &w);   // warm-up, checked below
    std::string rwrong;
    if (rrc == 0) {
      std::vector<float> got((size_t)n);
      pmpp::download(C, got.data(), sz);
      rwrong = p.compareWith(got.data());
    }
    for (int r = 0; r < a.reps && rrc == 0 && rwrong.empty(); ++r) {
      idleDevice();
      rrc = callResident(A, B, C, n, &w);
      resident_s = std::min(resident_s, w);
    }
#if defined(PMPP_BACKEND_METAL)
    double resident_gpu_s = 1e300;
    driverDispatch(A, B, C, n);
    for (int r = 0; r < a.reps; ++r) resident_gpu_s = std::min(resident_gpu_s, driverDispatch(A, B, C, n).gpu_s);
#endif
    const std::string rerr = pmpp::take_error();
    pmpp::release(A); pmpp::release(B); pmpp::release(C);
    if (rrc != 0 || !rwrong.empty() || !rerr.empty()) {
      std::printf("\nn = %d  stopped: the resident path failed (%s)\n", n,
                  rrc != 0 ? "non-zero status" : !rwrong.empty() ? rwrong.c_str() : rerr.c_str());
      return 1;
    }

    // The host-array path: your vecAddTimed.
    std::string wrong;
    Call timed[2];
    bool stop = false;
    const std::vector<int> modes = syncModes();
    for (size_t k = 0; k < modes.size() && !stop; ++k) {
      timed[k] = bestCall(p, modes[k], a.reps, &wrong);
      if (timed[k].rc != 0) {
        std::printf("\nn = %d  stopped: vecAddTimed returned %d (an allocation of %.0f MB may have failed)\n", n, timed[k].rc, bytes / 1e6);
        stop = true;
      } else if (!wrong.empty()) {
        std::printf("\nn = %d  stopped: %s\n", n, wrong.c_str());
        return 1;
      }
    }
    if (stop) break;

#if defined(PMPP_BACKEND_METAL)
    const Call& c = timed[0];
    const double total = phaseSum(c);
    std::printf("\nn = %d\n", n);
    std::printf("  host arrays in and out (your vecAddTimed)\n");
    std::printf("    %-28s %10.4f ms\n", "create buffers and copy in", ms(c.phase[0]));
    std::printf("    %-28s %10.4f ms\n", "encode and commit", ms(c.phase[1]));
    std::printf("    %-28s %10.4f ms   of which the command buffer reports %.4f ms on the GPU (%.2f%% of the total)\n", "wait",
                ms(c.phase[2]), ms(c.phase[4]), total > 0 ? 100.0 * c.phase[4] / total : 0.0);
    std::printf("    %-28s %10.4f ms\n", "copy out", ms(c.phase[3]));
    std::printf("    %-28s %10.4f ms   whole call: %.4f ms\n", "total of your four phases", ms(total), ms(c.wall_s));
    std::printf("  data already in shared buffers (your vecAddResident, timed by the driver)\n");
    std::printf("    %-28s %10.4f ms   the same dispatch issued by the driver reports %.4f ms on the GPU\n", "whole call",
                ms(resident_s), ms(resident_gpu_s));
    std::printf("    %-28s %10d      (the host-array path copies %.0f: two arrays in, one out)\n", "bytes copied", 0, 3 * bytes);
    std::printf("  %-30s %10.4f ms\n", "CPU loop", ms(cpu_s));
    pmpp::bench_row(a, "phases").n("n", n).n("wrap_s", c.phase[0]).n("encode_s", c.phase[1]).n("wait_s", c.phase[2])
        .n("readback_s", c.phase[3]).n("gpu_s", c.phase[4]).n("total_s", total).emit();
    // bytes_copied counts what the code path copies between host arrays and
    // buffers. It is a property of the code, not a measurement.
    pmpp::bench_row(a, "resident").n("n", n).n("seconds", resident_s).n("gpu_s", resident_gpu_s).n("bytes_copied", 0).emit();
#else
    std::printf("\nn = %d, blocks of 256 threads\n", n);
    std::printf("  %-22s %11s %11s %11s %11s %11s %11s %8s %11s\n", "", "alloc", "copies in", "kernel", "copy back", "free",
                "total", "kernel %", "whole call");
    for (size_t k = 0; k < modes.size(); ++k) {
      const Call& c = timed[k];
      const double total = phaseSum(c);
      const double share = total > 0 ? 100.0 * c.phase[2] / total : 0;
      std::printf("  %-22s %8.4f ms %8.4f ms %8.4f ms %8.4f ms %8.4f ms %8.4f ms %7.2f%% %8.4f ms\n",
                  modes[k] ? "sync after launch" : "no sync after launch", ms(c.phase[0]), ms(c.phase[1]), ms(c.phase[2]),
                  ms(c.phase[3]), ms(c.phase[4]), ms(total), share, ms(c.wall_s));
      pmpp::Json row = pmpp::bench_row(a, "phases");
      row.n("n", n).n("sync", modes[k]).n("alloc_s", c.phase[0]).n("h2d_s", c.phase[1]).n("kernel_s", c.phase[2])
         .n("d2h_s", c.phase[3]).n("free_s", c.phase[4]).n("total_s", total).n("kernel_share_pct", share);
      if (!modes[k] && timed[0].phase[2] > 0) row.n("nosync_kernel_pct", 100.0 * c.phase[2] / timed[0].phase[2]);
      row.emit();
    }
    std::printf("  the same work, three ways:\n");
    std::printf("    %-62s %10.4f ms\n", "your kernel phase, sync after launch", ms(timed[0].phase[2]));
    std::printf("    %-62s %10.4f ms\n", "launchVecAdd on device arrays, launch + synchronise (driver)", ms(resident_s));
    if (timed[0].phase[2] > 0)
      std::printf("    %-62s %10.4f ms   %.2f%% of the first line\n", "your kernel phase, no sync after launch", ms(timed[1].phase[2]),
                  100.0 * timed[1].phase[2] / timed[0].phase[2]);
    std::printf("  %-64s %10.4f ms\n", "CPU loop", ms(cpu_s));
    pmpp::bench_row(a, "resident").n("n", n).n("seconds", resident_s).emit();
#endif
    pmpp::bench_row(a, "cpu").n("n", n).n("seconds", cpu_s).emit();
  }
#if defined(PMPP_BACKEND_METAL)
  std::printf("\nCompare the resident whole call with the total of the host-array path, and both with what you wrote down.\n");
  std::printf("The gap between 'total of your four phases' and 'whole call' is time no clock inside vecAddTimed can see.\n");
#else
  std::printf("\nYour kernel phase with the synchronise and the driver's launchVecAdd time measure the same work.\n");
  std::printf("If they differ by much, a synchronise is missing or misplaced. Compare 'kernel %%' with what you wrote down.\n");
#endif
  std::printf("The CPU loop is one thread, compiled with this build's optimisation flags.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
