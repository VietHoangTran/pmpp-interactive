// Lab 23.1 driver: tests and benchmarks hiding copies behind a kernel.
// You do not need to edit this file; your work goes in work/.
//
// Every test hands your function a job whose input the driver wrote and whose
// output starts as NaN, compares the output with a CPU evaluation of the same
// polynomial, checks that the input was left alone, and then reads what the
// provided launcher recorded: which elements each launch covered, in which
// stream (CUDA family) or which buffers (Metal). On the emulator it also counts
// the heap bytes your call left allocated.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <map>
#include <set>
#include <utility>

#include "pmpp/harness.hpp"
#include "overlap.h"

#if defined(PMPP_BACKEND_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pmpp/metal.h"
#endif

// On the emulator, AddressSanitizer keeps a count of live heap bytes. A stream is
// a heap object there, so a stream created and never destroyed shows up in it.
// (LeakSanitizer would report the same at exit, but it does not run on every
// platform: on the Mac this lab was built on, it reports "not supported".)
#if defined(PMPP_BACKEND_EMU) && defined(__has_feature)
#if __has_feature(address_sanitizer)
#define LAB_HEAP_COUNT 1
#endif
#endif
#if defined(PMPP_BACKEND_EMU) && !defined(LAB_HEAP_COUNT) && defined(__SANITIZE_ADDRESS__)
#define LAB_HEAP_COUNT 1
#endif
#if defined(LAB_HEAP_COUNT)
#include <sanitizer/allocator_interface.h>
#endif

static std::string fmt(const char* f, ...) {
  char b[640];
  va_list ap;
  va_start(ap, f);
  std::vsnprintf(b, sizeof b, f, ap);
  va_end(ap);
  return b;
}

// ---- data and reference ---------------------------------------------------------
// Inputs in [-0.75, 0.75]: the Horner recurrence then shrinks rounding errors at
// every step, and every output stays below 2 in magnitude at any degree.
static std::vector<float> inputs(int n, uint32_t seed) {
  std::vector<float> x = pmpp::random_floats((size_t)n, seed);
  for (float& v : x) v *= 0.75f;
  return x;
}
static double hornerAt(double x, int degree) {
  double y = polyCoefficient(degree);
  for (int k = degree - 1; k >= 0; --k) y = y * x + polyCoefficient(k);
  return y;
}
static std::vector<float> reference(const std::vector<float>& x, int degree) {
  std::vector<float> y(x.size());
  for (size_t i = 0; i < x.size(); ++i) y[i] = (float)hornerAt(x[i], degree);
  return y;
}

// The chunk rule of overlap.h: the non-empty chunks, as (begin, count), in order.
static std::vector<std::pair<int, int>> chunkList(int n, int chunks) {
  std::vector<std::pair<int, int>> list;
  for (int k = 0; k < chunks; ++k) {
    const int b = (int)((long long)k * n / chunks), e = (int)((long long)(k + 1) * n / chunks);
    if (e > b) list.push_back({b, e - b});
  }
  return list;
}
static int chunkOf(const std::vector<std::pair<int, int>>& list, size_t i) {
  for (size_t j = 0; j < list.size(); ++j)
    if (i >= (size_t)list[j].first && i < (size_t)(list[j].first + list[j].second)) return (int)j;
  return -1;
}

static std::string checkOutput(const float* got, const std::vector<float>& ref, const std::vector<std::pair<int, int>>& list) {
  const pmpp::Cmp c = pmpp::compare(got, ref.data(), nullptr, ref.size());
  if (c.ok) return "";
  const size_t i = c.first_bad;
  std::string where = list.empty() ? "" : fmt(" (in chunk %d of those queued)", chunkOf(list, i));
  if (std::isnan(got[i]))
    return fmt("%zu of %zu outputs wrong; output %zu%s is still NaN, as the driver left it: nothing was copied back there",
               c.bad, ref.size(), i, where.c_str());
  return fmt("%zu of %zu outputs wrong; first at element %zu%s: got %.7g, expected %.7g", c.bad, ref.size(), i,
             where.c_str(), (double)got[i], (double)ref[i]);
}
static std::string checkInputKept(const float* hostIn, const std::vector<float>& x) {
  for (size_t i = 0; i < x.size(); ++i)
    if (std::memcmp(&hostIn[i], &x[i], sizeof(float)) != 0)
      return fmt("hostIn changed at element %zu: it is input only", i);
  return "";
}
static void fillNaN(float* p, size_t n) {
  std::fill(p, p + n, std::numeric_limits<float>::quiet_NaN());
}

static const int kTestDegree = 7;

// Printed before the cases where 64 * n passes 2^31, so that a crash there is not a mystery.
static void overflowNotice(int n, int chunks) {
  if ((long long)chunks * n > 2147483647LL)
    std::printf("  next: n = %d with %d chunks, where k*n passes 2^31. If the run stops here, look for k*n "
                "computed in int.\n", n, chunks);
  std::fflush(stdout);   // a crash would lose it from the buffer
}
// (streams or buffers, chunks). Metal uses the chunk counts with K = 1..4.
static const std::vector<std::pair<int, int>> kPairs = {{1, 1}, {1, 4}, {2, 2}, {3, 5}, {4, 16}, {8, 3}};

// ---- the benchmark's size, also the extra test size on real devices --------------
// As many floats as fit in a sixteenth of the memory the device reports, and never
// more than 64 Mi floats (256 MiB per array). The cap is a parameter of this lab,
// chosen to keep a run short; it is not a property of any device.
static const double kCapFloats = 64.0 * 1024 * 1024;
#if defined(PMPP_BACKEND_METAL)
static double deviceMemory() {
  const double a = pmpp::device_prop("maxBufferLength", 0), b = pmpp::device_prop("recommendedMaxWorkingSetSize", 0);
  if (a <= 0) return b;
  if (b <= 0) return a;
  return std::min(a, b);
}
static const char* memoryName() { return "the smaller of maxBufferLength and recommendedMaxWorkingSetSize"; }
#else
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* memoryName() { return "totalGlobalMem"; }
#endif
static int benchFloats() {
  const double mem = deviceMemory();
  const double fit = mem > 0 ? std::floor(mem / 16.0 / sizeof(float)) : kCapFloats;
  return (int)std::min(kCapFloats, fit);
}

// =================================================================================
#if !defined(PMPP_BACKEND_METAL)
// ---- CUDA, HIP and the emulator -------------------------------------------------
struct Arrays {   // one shape's buffers
  int n = 0;
  pmpp::Buf *din = nullptr, *dout = nullptr;
  float *pinIn = nullptr, *pinOut = nullptr;
  std::vector<float> pageIn, pageOut;
  bool ok = false;
  explicit Arrays(int n_) : n(n_) {
    din = pmpp::alloc((size_t)n * sizeof(float));
    dout = pmpp::alloc((size_t)n * sizeof(float));
    pinIn = pinnedAlloc((size_t)n);
    pinOut = pinnedAlloc((size_t)n);
    pageIn.assign((size_t)n, 0.0f);
    pageOut.assign((size_t)n, 0.0f);
    ok = pinIn && pinOut && pmpp::take_error().empty();
  }
  ~Arrays() { pmpp::release(din); pmpp::release(dout); pinnedFree(pinIn); pinnedFree(pinOut); }
  StagedJob job(bool pinned, int degree) {
    return StagedJob{pinned ? pinIn : pageIn.data(), pinned ? pinOut : pageOut.data(),
                     (float*)pmpp::raw(din), (float*)pmpp::raw(dout), n, degree};
  }
};

static const char* orderName(Order o) { return o == Order::Depth ? "depth" : "breadth"; }

// What launchPoly recorded, against the chunk rule.
static std::string checkLaunches(const StagedJob& job, int streams, int chunks) {
  const std::vector<std::pair<int, int>> list = chunkList(job.n, chunks);
  const std::vector<PolyCall>& calls = polyRecord().calls;
  for (size_t i = 0; i < calls.size(); ++i)
    if (calls[i].refused)
      return fmt("launch %zu was refused by launchPoly (%s): skip empty chunks", i, calls[i].refused);
  if (calls.empty()) return "launchPoly was never called";

  struct L { long long begin; int count; const void* stream; size_t issued; };
  std::vector<L> ls;
  for (size_t i = 0; i < calls.size(); ++i) {
    const PolyCall& c = calls[i];
    const long long inOff = (const float*)c.in - job.d_in, outOff = (const float*)c.out - job.d_out;
    if (inOff < 0 || inOff + c.count > job.n)
      return fmt("launch %zu read elements [%lld, %lld) of d_in, which holds %d", i, inOff, inOff + c.count, job.n);
    if (outOff != inOff)
      return fmt("launch %zu read from d_in + %lld but wrote to d_out + %lld: a chunk's input and output start at "
                 "the same element", i, inOff, outOff);
    if (c.degree != job.degree) return fmt("launch %zu was given degree %d; the job's is %d", i, c.degree, job.degree);
    if (c.stream == nullptr)
      return fmt("launch %zu (elements [%lld, %lld)) went into the default stream: pass the chunk's stream", i, inOff,
                 inOff + c.count);
    ls.push_back({inOff, c.count, c.stream, i});
  }
  std::sort(ls.begin(), ls.end(), [](const L& a, const L& b) { return a.begin < b.begin; });

  // (2a) Every element exactly once, then the chunk boundaries of the rule.
  long long next = 0;
  for (const L& l : ls) {
    if (l.begin < next)
      return fmt("elements [%lld, %lld) were computed twice", l.begin, std::min(next, l.begin + l.count));
    if (l.begin > next) return fmt("elements [%lld, %lld) were never computed", next, l.begin);
    next = l.begin + l.count;
  }
  if (next < job.n) return fmt("elements [%lld, %d) were never computed", next, job.n);
  if (ls.size() != list.size())
    return fmt("%zu launches for %zu non-empty chunks: chunk k is [k*n/chunks, (k+1)*n/chunks), one launch each",
               ls.size(), list.size());
  for (size_t j = 0; j < ls.size(); ++j)
    if (ls[j].begin != list[j].first || ls[j].count != list[j].second)
      return fmt("a launch covered [%lld, %lld), but chunk %zu of the rule is [%d, %d)", ls[j].begin,
                 ls[j].begin + ls[j].count, j, list[j].first, list[j].first + list[j].second);

  // (2b) Streams.
  std::set<const void*> distinct;
  for (const L& l : ls) distinct.insert(l.stream);
  const size_t want = std::min((size_t)streams, list.size());
  if (distinct.size() != want)
    return fmt("the launches used %zu distinct stream%s; with %d stream%s and %zu non-empty chunk%s they should use %zu "
               "(chunk j goes to stream j mod streams)", distinct.size(), distinct.size() == 1 ? "" : "s", streams,
               streams == 1 ? "" : "s", list.size(), list.size() == 1 ? "" : "s", want);
  if (streams >= 2)
    for (size_t j = 0; j + 1 < ls.size(); ++j)
      if (ls[j].stream == ls[j + 1].stream)
        return fmt("chunks %zu and %zu went into the same stream; with %d streams, neighbours should not "
                   "(chunk j goes to stream j mod streams)", j, j + 1, streams);
  return "";
}

static std::string stagedCase(Arrays& A, const std::vector<float>& x, const std::vector<float>& ref, bool pinned,
                              int streams, int chunks, Order order) {
  StagedJob job = A.job(pinned, kTestDegree);
  std::memcpy((float*)job.hostIn, x.data(), x.size() * sizeof(float));
  fillNaN(job.hostOut, x.size());
  polyResetRecord();
#if defined(LAB_HEAP_COUNT)
  const size_t heapBefore = __sanitizer_get_current_allocated_bytes();
#endif
  const int rc = runStaged(job, streams, chunks, order);
#if defined(LAB_HEAP_COUNT)
  const size_t heapAfter = __sanitizer_get_current_allocated_bytes();
#endif
  if (rc != 0) return fmt("runStaged returned %d; a call that works returns 0", rc);
  // (1) The output, and the input left alone.
  std::string wrong = checkOutput(job.hostOut, ref, chunkList(job.n, chunks));
  if (wrong.empty()) wrong = checkInputKept(job.hostIn, x);
  // (2) What launchPoly saw.
  if (wrong.empty()) wrong = checkLaunches(job, streams, chunks);
#if defined(LAB_HEAP_COUNT)
  // (4) Nothing left allocated.
  if (wrong.empty() && heapAfter > heapBefore)
    wrong = fmt("%zu heap bytes allocated during runStaged were still allocated when it returned "
                "(AddressSanitizer's count). In the emulator every stream is a heap object: did you destroy "
                "every stream you created?", heapAfter - heapBefore);
#endif
  return wrong;
}

// The provided baseline is checked too, quietly: a case appears only if it fails.
static std::string serialCheck(Arrays& A, const std::vector<float>& x, const std::vector<float>& ref) {
  StagedJob job = A.job(true, kTestDegree);
  std::memcpy((float*)job.hostIn, x.data(), x.size() * sizeof(float));
  fillNaN(job.hostOut, x.size());
  polyResetRecord();
  if (runSerial(job) != 0) return "the provided runSerial failed";
  return checkOutput(job.hostOut, ref, {});
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::vector<int> sizes = {1, 7, 1000, 4099, 65536};
  if (!pmpp::is_emu()) sizes.push_back(benchFloats());
  for (int n : sizes) {
    const std::vector<float> x = inputs(n, 2300u + (uint32_t)n);
    const std::vector<float> ref = reference(x, kTestDegree);
    Arrays A(n);
    if (!A.ok) {
      t.check(fmt("allocate the arrays for n = %d", n), [&] { return std::string("allocation failed (see above)"); });
      continue;
    }
    const std::string serial = serialCheck(A, x, ref);
    if (!serial.empty()) t.check(fmt("provided runSerial, n = %d (a driver self-check)", n), [&] { return serial; });

    std::vector<std::pair<int, int>> pairs = kPairs;
    if (n == 7) { pairs.push_back({2, 16}); pairs.push_back({3, 16}); }   // empty chunks between full ones
    if (!pmpp::is_emu() && n == sizes.back()) pairs.push_back({4, 64});   // 64 * n can pass 2^31 here
    for (const auto& p : pairs)
      for (Order order : {Order::Depth, Order::Breadth})
        for (bool pinned : {true, false}) {
          if (order == Order::Depth && pinned) overflowNotice(n, p.second);
          const std::string name = fmt("runStaged, n = %d, %d stream%s, %d chunk%s, %s, %s", n, p.first,
                                       p.first == 1 ? "" : "s", p.second, p.second == 1 ? "" : "s", orderName(order),
                                       pinned ? "pinned" : "pageable");
          t.check(name, [&] { return stagedCase(A, x, ref, pinned, p.first, p.second, order); });
        }
  }
  return t.finish();
}

// ---- benchmark --------------------------------------------------------------------
static void idleDevice() {   // waits for the device and clears the runtime's error state
  pmpp::run([](void*) {});
  pmpp::take_error();
}

// One untimed warm-up call, then `reps` timed calls; the warm-up's and the last
// call's outputs are checked. Returns the best time, or sets `wrong`.
static double timeCalls(int reps, float* out, size_t n, const std::function<int()>& call,
                        const std::function<std::string()>& check, std::string* wrong) {
  double best = std::numeric_limits<double>::infinity();
  for (int r = 0; r <= reps; ++r) {
    if (r == 0 || r == reps) fillNaN(out, n);
    idleDevice();
    polyResetRecord();
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = call();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rc != 0) { *wrong = fmt("returned %d", rc); return best; }
    if (r == 0 || r == reps) {
      *wrong = check();
      if (!wrong->empty()) return best;
    }
    if (r > 0) best = std::min(best, s);
  }
  return best;
}

static int bench(const pmpp::Args& a) {
  const int n = benchFloats();
  const double bytes = (double)n * sizeof(float);
  if (n < 4096) { std::printf("bench skipped: the device reports too little memory\n"); return 1; }
  std::printf("size: %d floats = %.0f bytes (%.1f MiB) per array\n", n, bytes, bytes / 1048576.0);
  std::printf("  the smaller of a 64 Mi-float cap (this lab's parameter, to keep a run short) and a sixteenth of "
              "%s (%.0f bytes)\n", memoryName(), deviceMemory());

  int engines = -1, concurrent = -1;
  probeEngines(&engines, &concurrent);
#if defined(PMPP_BACKEND_HIP)
  std::printf("asyncEngineCount %d, concurrentKernels %d. On ROCm asyncEngineCount is a runtime constant, the same "
              "on every device: it counts no copy hardware (ch23-facts B3)\n", engines, concurrent);
#else
  std::printf("asyncEngineCount %d (copy engines: with 1, copies in and out take turns; with 2 they can overlap, "
              "ch23-facts A7), concurrentKernels %d\n", engines, concurrent);
#endif

  Arrays A(n);
  if (!A.ok) { std::printf("bench skipped: could not allocate the arrays\n"); return 1; }
  const std::vector<float> x = inputs(n, 777u);
  std::memcpy(A.pinIn, x.data(), bytes);
  std::memcpy(A.pageIn.data(), x.data(), bytes);

  // Calibration: the degree whose kernel time is closest to the copy in.
  StagedJob pj = A.job(true, 0);
  double h2d = 0, d2h = 0;
  if (probeCopyIn(pj, a.reps, &h2d) != 0) return 1;
  std::printf("\ncalibration: copy in of the whole array from pinned memory %.6f s\n", h2d);
  std::map<int, double> kt;   // degree -> kernel seconds
  auto measure = [&](int d) -> bool {
    double s = 0;
    if (probeKernel(pj, d, a.reps, &s) != 0) return false;
    kt[d] = s;
    std::printf("  kernel at degree %6d: %.6f s\n", d, s);
    return true;
  };
  int d = 1;
  for (;; d *= 2) {
    if (!measure(d)) return 1;
    if (kt[d] >= 2 * h2d || d >= (1 << 16)) break;
  }
  // Between the last two powers of two the time grows about linearly with the
  // degree once the arithmetic dominates; try the degree that line says.
  if (d > 1 && kt[d] > h2d && kt[d / 2] < h2d) {
    const int lo = d / 2;
    const int guess = lo + (int)std::lround((h2d - kt[lo]) / (kt[d] - kt[lo]) * (d - lo));
    if (guess > lo && guess < d && !measure(guess)) return 1;
  }
  int degree = kt.begin()->first;
  for (const auto& p : kt)
    if (std::fabs(p.second - h2d) < std::fabs(kt[degree] - h2d)) degree = p.first;
  std::printf("chosen degree %d: kernel %.6f s against copy in %.6f s (measured here, not assumed)\n", degree,
              kt[degree], h2d);
  pj.degree = degree;
  if (probeCopyOut(pj, a.reps, &d2h) != 0) return 1;
  std::printf("\nparts, each alone over the whole array: copy in %.6f s, kernel %.6f s, copy out %.6f s\n", h2d,
              kt[degree], d2h);
  pmpp::bench_row(a, "parts").n("h2d_s", h2d).n("kernel_s", kt[degree]).n("d2h_s", d2h).n("bytes", bytes)
      .n("degree", degree).n("async_engines", engines).n("concurrent_kernels", concurrent).emit();

  // The reference for every row: the provided serial path's output, itself
  // checked against the CPU on a sample (all n at this degree would take long).
  StagedJob sj = A.job(true, degree);
  fillNaN(sj.hostOut, (size_t)n);
  if (runSerial(sj) != 0) { std::printf("bench: the provided runSerial failed\n"); return 1; }
  std::vector<float> ref(sj.hostOut, sj.hostOut + n);
  const size_t stride = std::max<size_t>(1, (size_t)n / 65536);
  for (size_t i = 0; i < (size_t)n; i += stride) {
    const double r = hornerAt(x[i], degree);
    if (!(std::fabs(ref[i] - r) <= 1e-4 * (1 + std::fabs(r)))) {
      std::printf("bench: runSerial's output at %zu is %.7g, the CPU says %.7g\n", i, (double)ref[i], r);
      return 1;
    }
  }

  std::printf("\nEach time is one whole call (stream creation included), best of %d after one untimed warm-up.\n",
              a.reps);
  std::printf("The warm-up's and the last call's outputs are checked; a wrong one is reported without a time.\n\n");
  std::printf("%-8s %7s %6s %-7s %-8s %12s\n", "row", "streams", "chunks", "order", "host", "seconds");
  std::map<std::string, double> secs;
  int rc_all = 0;
  auto check = [&](const float* out) { return [&ref, out] { return checkOutput(out, ref, {}); }; };

  for (bool pinned : {true, false}) {
    StagedJob job = A.job(pinned, degree);
    std::string wrong;
    const double s = timeCalls(a.reps, job.hostOut, (size_t)n, [&] { return runSerial(job); }, check(job.hostOut), &wrong);
    const char* host = pinned ? "pinned" : "pageable";
    if (!wrong.empty()) { std::printf("%-8s %7s %6s %-7s %-8s  failed: runSerial %s\n", "serial", "", "", "", host, wrong.c_str()); rc_all = 1; continue; }
    secs[std::string("serial/") + host] = s;
    std::printf("%-8s %7s %6s %-7s %-8s %12.6f\n", "serial", "", "", "", host, s);
    pmpp::bench_row(a, "serial").s("host", host).n("seconds", s).n("bytes", bytes).n("degree", degree).emit();
  }

  struct Config { int streams, chunks; Order order; bool pinned; };
  std::vector<Config> configs;
  for (int st : {1, 2, 3, 4, 8})
    for (int ch : {4, 16, 64})
      for (Order o : {Order::Depth, Order::Breadth}) configs.push_back({st, ch, o, true});
  configs.push_back({4, 16, Order::Depth, false});
  for (const Config& c : configs) {
    StagedJob job = A.job(c.pinned, degree);
    std::string wrong;
    const double s = timeCalls(a.reps, job.hostOut, (size_t)n,
                               [&] { return runStaged(job, c.streams, c.chunks, c.order); }, check(job.hostOut), &wrong);
    const char* host = c.pinned ? "pinned" : "pageable";
    if (!wrong.empty()) {
      std::printf("%-8s %7d %6d %-7s %-8s  failed: runStaged %s\n", "staged", c.streams, c.chunks, orderName(c.order),
                  host, wrong.c_str());
      rc_all = 1;
      continue;
    }
    secs[fmt("staged/%d/%d/%s/%s", c.streams, c.chunks, orderName(c.order), host)] = s;
    std::printf("%-8s %7d %6d %-7s %-8s %12.6f\n", "staged", c.streams, c.chunks, orderName(c.order), host, s);
    pmpp::bench_row(a, "staged").n("streams", c.streams).n("chunks", c.chunks).s("order", orderName(c.order))
        .s("host", host).n("seconds", s).n("bytes", bytes).n("degree", degree).emit();
  }

  auto ratio = [&](const std::string& p, const std::string& q) -> double {
    auto i = secs.find(p), j = secs.find(q);
    return (i == secs.end() || j == secs.end()) ? std::numeric_limits<double>::quiet_NaN() : i->second / j->second;
  };
  std::printf("\nRatios of whole-call times, to set beside what you wrote down:\n");
  std::printf("  (1) serial / staged, 4 streams, 16 chunks, depth, pinned    = %.3f\n",
              ratio("serial/pinned", "staged/4/16/depth/pinned"));
  std::printf("  (2) serial / staged, 4 streams, 16 chunks, depth, pageable  = %.3f\n",
              ratio("serial/pageable", "staged/4/16/depth/pageable"));
  std::printf("  (4) breadth / depth, pinned:");
  for (int st : {1, 2, 4, 8}) std::printf("  %d streams, 16 chunks %.3f;", st,
                                          ratio(fmt("staged/%d/16/breadth/pinned", st), fmt("staged/%d/16/depth/pinned", st)));
  std::printf("\nThese are measurements of this machine at one size and one degree, not properties of the method.\n");
  return rc_all;
}

// =================================================================================
#else
// ---- Metal ----------------------------------------------------------------------
struct MetalSetup {
  id<MTLDevice> device = nil;
  id<MTLCommandQueue> queue = nil;
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
  if (!pmpp::metal::pipeline(PMPP_METAL_SOURCE, "poly_kernel")) {
    s.error = "kernel.metal did not build into a pipeline: " + pmpp::take_error();
    if (s.error.size() > 600) s.error.resize(600);
  }
  return s;
}

// The pool is inside the call on purpose: what the call leaves to it is released
// when it drains.
static int callInFlight(const MetalJob& job, int buffers, int chunks) {
  int rc;
  @autoreleasepool { rc = runInFlight(job, buffers, chunks); }
  return rc;
}
static int callSerial(const MetalJob& job) {
  int rc;
  @autoreleasepool { rc = runSerialMetal(job); }
  return rc;
}

// What encodePoly and waitFor saw, against the chunk rule.
static std::string checkDispatches(const MetalJob& job, int K, int chunks) {
  const std::vector<std::pair<int, int>> list = chunkList(job.n, chunks);
  const PolyRecord& rec = polyRecord();
  const std::vector<PolyCall>& calls = rec.calls;
  for (size_t i = 0; i < calls.size(); ++i)
    if (calls[i].refused) return fmt("encodePoly call %zu was refused (%s)", i, calls[i].refused);
  if (calls.size() != list.size())
    return fmt("%zu encodePoly calls for %zu non-empty chunks: one per chunk, and none for an empty one", calls.size(),
               list.size());
  for (size_t j = 0; j < calls.size(); ++j) {
    if (calls[j].count != list[j].second)
      return fmt("dispatch %zu covered %d elements, but chunk %zu of the rule, [%d, %d), has %d", j, calls[j].count, j,
                 list[j].first, list[j].first + list[j].second, list[j].second);
    if (calls[j].degree != job.degree) return fmt("dispatch %zu was given degree %d; the job's is %d", j, calls[j].degree, job.degree);
  }
  std::set<const void*> inBufs;
  for (const PolyCall& c : calls) inBufs.insert(c.in);
  if (inBufs.size() > (size_t)K)
    return fmt("the dispatches read %zu different input buffers; with %d buffers in flight there are %d slots, reused",
               inBufs.size(), K, K);
  if (K >= 2)
    for (size_t j = 0; j + 1 < calls.size(); ++j)
      if (calls[j].in == calls[j + 1].in)
        return fmt("chunks %zu and %zu used the same input buffer; with %d slots, chunk j uses slot j mod %d", j, j + 1, K, K);
  if (rec.waits != (int)calls.size())
    return fmt("waitFor was called %d times for %zu command buffers: wait for each one with waitFor, which checks it "
               "and adds up its GPU time", rec.waits, calls.size());
  for (size_t j = 0; j < calls.size(); ++j) {
    const int need = std::max(0, (int)j - K + 1);
    if (calls[j].waitsBefore > need)
      return fmt("chunk %zu was encoded after %d wait%s; with %d slots only %d command buffer%s need to have finished "
                 "by then. Waiting for more leaves the GPU idle while the CPU fills", j, calls[j].waitsBefore,
                 calls[j].waitsBefore == 1 ? "" : "s", K, need, need == 1 ? "" : "s");
  }
  return "";
}

static std::string inFlightCase(const std::vector<float>& x, const std::vector<float>& ref, std::vector<float>& in,
                                std::vector<float>& out, int K, int chunks) {
  MetalSetup& m = metalSetup();
  std::memcpy(in.data(), x.data(), x.size() * sizeof(float));
  fillNaN(out.data(), out.size());
  MetalJob job{m.device, m.queue, in.data(), out.data(), (int)x.size(), kTestDegree};
  polyResetRecord();
  const int rc = callInFlight(job, K, chunks);
  if (rc != 0) return fmt("runInFlight returned %d; a call that works returns 0", rc);
  std::string wrong = checkOutput(out.data(), ref, chunkList(job.n, chunks));
  if (wrong.empty()) wrong = checkInputKept(in.data(), x);
  if (wrong.empty()) wrong = checkDispatches(job, K, chunks);
  return wrong;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string broken = metalSetup().error;
  if (!broken.empty()) {
    t.check("kernel.metal builds into a pipeline", [&] { return broken; });
    return t.finish();
  }
  std::vector<int> sizes = {1, 7, 1000, 4099, 65536, benchFloats()};
  for (int n : sizes) {
    const std::vector<float> x = inputs(n, 2300u + (uint32_t)n);
    const std::vector<float> ref = reference(x, kTestDegree);
    std::vector<float> in((size_t)n), out((size_t)n);
    {   // the provided baseline, quietly: a case appears only if it fails
      MetalSetup& m = metalSetup();
      std::memcpy(in.data(), x.data(), x.size() * sizeof(float));
      fillNaN(out.data(), out.size());
      MetalJob job{m.device, m.queue, in.data(), out.data(), n, kTestDegree};
      polyResetRecord();
      std::string serial = callSerial(job) != 0 ? "the provided runSerialMetal failed" : checkOutput(out.data(), ref, {});
      if (!serial.empty()) t.check(fmt("provided runSerialMetal, n = %d (a driver self-check)", n), [&] { return serial; });
    }
    std::vector<int> chunkCounts;
    for (const auto& p : kPairs) chunkCounts.push_back(p.second);
    if (n == sizes.back()) chunkCounts.push_back(64);   // the benchmark's size: 64 * n can pass 2^31
    for (int K : {1, 2, 3, 4})
      for (int chunks : chunkCounts) {
        if (K == 1) overflowNotice(n, chunks);
        const std::string name = fmt("runInFlight, n = %d, %d buffer%s, %d chunk%s", n, K, K == 1 ? "" : "s", chunks,
                                     chunks == 1 ? "" : "s");
        t.check(name, [&] { return inFlightCase(x, ref, in, out, K, chunks); });
      }
  }
  return t.finish();
}

// ---- benchmark --------------------------------------------------------------------
struct Measured { double best; double gpu; bool overlap; };

// Did the CPU encode a chunk (just after filling it) while the GPU was running
// another chunk's command buffer? Both clocks are mach time (ch05-facts C9).
static bool fillOverlapped(const PolyRecord& rec) {
  for (const PolyCall& c : rec.calls)
    for (size_t i = 0; i < rec.gpuStart.size(); ++i)
      if (c.hostTime > rec.gpuStart[i] && c.hostTime < rec.gpuEnd[i]) return true;
  return false;
}

static Measured timeCalls(int reps, std::vector<float>& out, const std::function<int()>& call,
                          const std::function<std::string()>& check, std::string* wrong) {
  Measured m{std::numeric_limits<double>::infinity(), 0, false};
  for (int r = 0; r <= reps; ++r) {
    if (r == 0 || r == reps) fillNaN(out.data(), out.size());
    polyResetRecord();
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = call();
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    if (rc != 0) { *wrong = fmt("returned %d", rc); return m; }
    if (r == 0 || r == reps) {
      *wrong = check();
      if (!wrong->empty()) return m;
    }
    if (r > 0 && s < m.best) {
      m.best = s;
      m.gpu = polyRecord().gpuSeconds;
      m.overlap = fillOverlapped(polyRecord());
    }
  }
  return m;
}

static int bench(const pmpp::Args& a) {
  MetalSetup& m = metalSetup();
  if (!m.error.empty()) { std::printf("bench skipped: %s\n", m.error.c_str()); return 1; }
  const int n = benchFloats();
  const double bytes = (double)n * sizeof(float);
  if (n < 4096) { std::printf("bench skipped: the device reports too little memory\n"); return 1; }
  std::printf("size: %d floats = %.0f bytes (%.1f MiB) per array\n", n, bytes, bytes / 1048576.0);
  std::printf("  the smaller of a 64 Mi-float cap (this lab's parameter, to keep a run short) and a sixteenth of "
              "%s (%.0f bytes)\n", memoryName(), deviceMemory());
  std::printf("No copy engine is documented for Apple silicon (ch23-facts C4): the parts here are the CPU's fill, the "
              "GPU's kernel and the CPU's drain.\n");

  const std::vector<float> x = inputs(n, 777u);
  std::vector<float> hostIn(x), hostOut((size_t)n);
  int rc_all = 0;
  double fill = 0, drain = 0;
  std::map<int, double> kt;
  int degree = 1;
  @autoreleasepool {
    id<MTLBuffer> bin = [m.device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    id<MTLBuffer> bout = [m.device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
    if (!bin || !bout) { std::printf("bench skipped: could not make two shared buffers of the array's size\n"); return 1; }
    // The CPU's parts, best of reps after a warm-up.
    fill = drain = std::numeric_limits<double>::infinity();
    for (int r = 0; r <= a.reps; ++r) {
      auto t0 = std::chrono::steady_clock::now();
      std::memcpy(bin.contents, hostIn.data(), (size_t)bytes);
      auto t1 = std::chrono::steady_clock::now();
      std::memcpy(hostOut.data(), bout.contents, (size_t)bytes);
      auto t2 = std::chrono::steady_clock::now();
      if (r > 0) {
        fill = std::min(fill, std::chrono::duration<double>(t1 - t0).count());
        drain = std::min(drain, std::chrono::duration<double>(t2 - t1).count());
      }
    }
    // What a new buffer costs: making it, then the first fill of its pages.
    double create = std::numeric_limits<double>::infinity(), fresh = create;
    for (int r = 0; r <= a.reps; ++r) {
      @autoreleasepool {
        auto t0 = std::chrono::steady_clock::now();
        id<MTLBuffer> f = [m.device newBufferWithLength:(NSUInteger)bytes options:MTLResourceStorageModeShared];
        auto t1 = std::chrono::steady_clock::now();
        if (!f) { std::printf("bench skipped: could not make a third shared buffer of the array's size\n"); return 1; }
        std::memcpy(f.contents, hostIn.data(), (size_t)bytes);
        auto t2 = std::chrono::steady_clock::now();
        if (r > 0) {
          create = std::min(create, std::chrono::duration<double>(t1 - t0).count());
          fresh = std::min(fresh, std::chrono::duration<double>(t2 - t1).count());
        }
      }
    }
    std::printf("\nfirst touch, observed here: making a shared buffer of the array's size %.6f s; its first fill %.6f s,\n"
                "against %.6f s for a fill of a buffer already used. slotBuffer keeps runInFlight's slots between calls\n"
                "so that the rows below time the overlap, not this.\n", create, fresh, fill);
    pmpp::bench_row(a, "firsttouch").n("create_s", create).n("fresh_fill_s", fresh).n("fill_s", fill).n("bytes", bytes).emit();

    // The GPU's part at one degree, timed exactly as the serial row is: the provided
    // runSerialMetal at that degree, and the GPU's own account of its command buffer.
    // (A lone command buffer timed apart from the fill and drain gave a GPU time well
    // above the serial row's at the same degree on the machine this lab was built on.)
    auto measure = [&](int d) -> bool {
      MetalJob jd{m.device, m.queue, hostIn.data(), hostOut.data(), n, d};
      double best = std::numeric_limits<double>::infinity();
      for (int r = 0; r <= a.reps; ++r) {
        polyResetRecord();
        if (callSerial(jd) != 0) return false;
        if (r > 0) best = std::min(best, polyRecord().gpuSeconds);
      }
      kt[d] = best;
      std::printf("  kernel at degree %6d: %.6f s (GPU time inside runSerialMetal)\n", d, best);
      return true;
    };
    // The CPU does both the fill and the drain, one after the other on one thread, so
    // its lane is fill + drain. The kernel is matched to that sum, the pipeline's best case.
    const double cpu = fill + drain;
    std::printf("\ncalibration: the CPU's lane, fill + drain of the whole array, %.6f s + %.6f s = %.6f s\n", fill,
                drain, cpu);
    int d = 1;
    for (;; d *= 2) {
      if (!measure(d)) return 1;
      if (kt[d] >= 2 * cpu || d >= (1 << 16)) break;
    }
    if (d > 1 && kt[d] > cpu && kt[d / 2] < cpu) {
      const int lo = d / 2;
      const int guess = lo + (int)std::lround((cpu - kt[lo]) / (kt[d] - kt[lo]) * (d - lo));
      if (guess > lo && guess < d && !measure(guess)) return 1;
    }
    degree = kt.begin()->first;
    for (const auto& p : kt)
      if (std::fabs(p.second - cpu) < std::fabs(kt[degree] - cpu)) degree = p.first;
    std::printf("chosen degree %d: kernel %.6f s against fill + drain %.6f s (measured here, not assumed)\n", degree,
                kt[degree], cpu);
  }
  std::printf("\nparts, each alone over the whole array: fill %.6f s, kernel %.6f s, drain %.6f s\n", fill, kt[degree],
              drain);
  pmpp::bench_row(a, "parts").n("fill_s", fill).n("kernel_s", kt[degree]).n("drain_s", drain).n("bytes", bytes)
      .n("degree", degree).emit();

  // The reference: the provided serial path's output, checked against the CPU on a sample.
  MetalJob job{m.device, m.queue, hostIn.data(), hostOut.data(), n, degree};
  fillNaN(hostOut.data(), hostOut.size());
  polyResetRecord();
  if (callSerial(job) != 0) { std::printf("bench: the provided runSerialMetal failed\n"); return 1; }
  const std::vector<float> ref(hostOut);
  const size_t stride = std::max<size_t>(1, (size_t)n / 65536);
  for (size_t i = 0; i < (size_t)n; i += stride) {
    const double r = hornerAt(x[i], degree);
    if (!(std::fabs(ref[i] - r) <= 1e-4 * (1 + std::fabs(r)))) {
      std::printf("bench: runSerialMetal's output at %zu is %.7g, the CPU says %.7g\n", i, (double)ref[i], r);
      return 1;
    }
  }
  auto check = [&] { return checkOutput(hostOut.data(), ref, {}); };

  std::printf("\nEach time is one whole call, best of %d after one untimed warm-up.\n", a.reps);
  std::printf("The warm-up's and the last call's outputs are checked; a wrong one is reported without a time.\n");
  std::printf("GPU s: the sum of GPUEndTime - GPUStartTime over the command buffers passed to waitFor.\n");
  std::printf("overlap: observed here, not documented: whether the CPU encoded a chunk, right after filling it, while\n"
              "the GPU was running another chunk's command buffer.\n\n");
  std::printf("%-8s %7s %6s %12s %12s %8s\n", "row", "buffers", "chunks", "seconds", "GPU s", "overlap");
  std::map<std::string, double> secs;
  {
    std::string wrong;
    const Measured s = timeCalls(a.reps, hostOut, [&] { return callSerial(job); }, check, &wrong);
    if (!wrong.empty()) { std::printf("%-8s  failed: runSerialMetal %s\n", "serial", wrong.c_str()); rc_all = 1; }
    else {
      secs["serial"] = s.best;
      std::printf("%-8s %7s %6s %12.6f %12.6f %8s\n", "serial", "", "", s.best, s.gpu, s.overlap ? "yes" : "no");
      pmpp::bench_row(a, "serial").s("host", "shared").n("seconds", s.best).n("gpu_s", s.gpu).n("bytes", bytes)
          .n("degree", degree).emit();
    }
  }
  for (int K : {1, 2, 3, 4})
    for (int chunks : {4, 16, 64}) {
      std::string wrong;
      const Measured s = timeCalls(a.reps, hostOut, [&] { return callInFlight(job, K, chunks); }, check, &wrong);
      if (!wrong.empty()) {
        std::printf("%-8s %7d %6d  failed: runInFlight %s\n", "inflight", K, chunks, wrong.c_str());
        rc_all = 1;
        continue;
      }
      secs[fmt("inflight/%d/%d", K, chunks)] = s.best;
      std::printf("%-8s %7d %6d %12.6f %12.6f %8s\n", "inflight", K, chunks, s.best, s.gpu, s.overlap ? "yes" : "no");
      pmpp::bench_row(a, "inflight").n("buffers", K).n("chunks", chunks).n("seconds", s.best).n("gpu_s", s.gpu)
          .b("overlap_observed", s.overlap).n("bytes", bytes).n("degree", degree).emit();
    }

  auto ratio = [&](const std::string& p, const std::string& q) -> double {
    auto i = secs.find(p), j = secs.find(q);
    return (i == secs.end() || j == secs.end()) ? std::numeric_limits<double>::quiet_NaN() : i->second / j->second;
  };
  std::printf("\nRatios of whole-call times, to set beside what you wrote down:\n");
  std::printf("  (3) in flight, 16 chunks: 1 buffer / 3 buffers = %.3f\n", ratio("inflight/1/16", "inflight/3/16"));
  std::printf("      serial / in flight with 3 buffers, 16 chunks = %.3f\n", ratio("serial", "inflight/3/16"));
  std::printf("These are measurements of this machine at one size and one degree, not properties of the method.\n");
  return rc_all;
}
#endif

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
