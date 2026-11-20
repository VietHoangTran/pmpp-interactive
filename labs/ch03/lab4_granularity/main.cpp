// Lab 3.4 driver: tests and times three ways to share one matrix multiplication
// among threads: one thread per element (provided), per row and per column (yours).
// You do not need to edit this file; your work goes in work/.
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// Both emulator flavours run on the CPU and get the small problems only.
#if defined(PMPP_BACKEND_EMU) || defined(PMPP_BACKEND_RACE)
static const bool kEmulated = true;
#else
static const bool kEmulated = false;
#endif

enum Mode { kElement = 0, kRow = 1, kCol = 2 };
static const char* const kModeName[3] = {"element", "row", "col"};
static const char* const kModeLabel[3] = {"one thread per element (provided)", "one thread per row", "one thread per column"};

#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeElement(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M, id<MTLBuffer> N,
                   id<MTLBuffer> P, uint32_t width);
void encodeRow(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M, id<MTLBuffer> N,
               id<MTLBuffer> P, uint32_t width);
void encodeCol(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> M, id<MTLBuffer> N,
               id<MTLBuffer> P, uint32_t width);
static id<MTLComputePipelineState> modePipeline(Mode m) {
  static const char* const fn[3] = {"element_matmul", "row_matmul", "col_matmul"};
  return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn[m]);
}
static pmpp::Timing runMode(Mode m, Buf* M, Buf* N, Buf* P, int w) {
  auto pso = modePipeline(m);
  if (!pso) return {0, 0};
  return pmpp::run([&](void* v) {
    id<MTLComputeCommandEncoder> enc = ((pmpp::metal::Pass*)v)->encoder;
    id<MTLBuffer> bm = pmpp::metal::buffer(M), bn = pmpp::metal::buffer(N), bp = pmpp::metal::buffer(P);
    if (m == kElement) encodeElement(enc, pso, bm, bn, bp, (uint32_t)w);
    else if (m == kRow) encodeRow(enc, pso, bm, bn, bp, (uint32_t)w);
    else encodeCol(enc, pso, bm, bn, bp, (uint32_t)w);
  });
}
// The lab fixes its threadgroups at 256 threads. Metal's limit belongs to the
// pipeline, not to the device, so ask each pipeline. Empty string = fine.
static std::string launchShapeProblem(Mode m) {
  auto pso = modePipeline(m);
  if (!pso || pso.maxTotalThreadsPerThreadgroup >= 256) return "";
  return "this pipeline allows " + std::to_string(pso.maxTotalThreadsPerThreadgroup) +
         " threads per threadgroup, fewer than the 256 the lab uses";
}
#else
void launchElement(float* M, float* N, float* P, int Width);
void launchRow(float* M, float* N, float* P, int Width);
void launchCol(float* M, float* N, float* P, int Width);
static pmpp::Timing runMode(Mode m, Buf* M, Buf* N, Buf* P, int w) {
  return pmpp::run([&](void*) {
    float *pm = (float*)pmpp::raw(M), *pn = (float*)pmpp::raw(N), *pp = (float*)pmpp::raw(P);
    if (m == kElement) launchElement(pm, pn, pp, w);
    else if (m == kRow) launchRow(pm, pn, pp, w);
    else launchCol(pm, pn, pp, w);
  });
}
static std::string launchShapeProblem(Mode) {
  const double most = pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (most >= 256) return "";
  return "this device allows " + std::to_string((int)most) + " threads per block, fewer than the 256 the lab uses";
}
#endif

// ---------- what the tests put around one multiplication ----------
// P is pre-filled with a value no product equals, so an element a kernel never
// writes is detected.
//   Emulator (backend emu): every buffer has exactly its size, and AddressSanitizer
//   stops the run at the first access outside it and names the line.
//   Everywhere else: nothing reports such an access, so every buffer gets slack after
//   its last element. The slack after M and N holds NaN (a kernel that reads it
//   produces NaN); the slack after P holds a canary (a kernel that writes it is caught).
static const float kUnwritten = 12345.0f;
static const float kCanary = -54321.0f;

struct Guarded {
  int w;
  size_t slack;   // floats after the last element of each buffer
  std::vector<float> m, n, ref, mag;
  Buf *M, *N, *P = nullptr;

  Guarded(int width, uint32_t seed) : w(width) {
    const size_t e = (size_t)w * w;
    slack = pmpp::is_emu() ? 0 : 256 * (size_t)w + 256;   // a whole surplus block of rows
    m = pmpp::random_floats(e, seed);
    n = pmpp::random_floats(e, seed + 1);
    pmpp::cpu_gemm(m.data(), n.data(), w, w, w, ref, &mag);
    M = padded(m, std::numeric_limits<float>::quiet_NaN());
    N = padded(n, std::numeric_limits<float>::quiet_NaN());
    P = pmpp::alloc((e + slack) * 4);
  }
  Guarded(const Guarded&) = delete;
  ~Guarded() { pmpp::release(M); pmpp::release(N); pmpp::release(P); }

  Buf* padded(const std::vector<float>& v, float fill) const {
    std::vector<float> host(v);
    host.resize(v.size() + slack, fill);
    Buf* b = pmpp::alloc(host.size() * 4);
    pmpp::upload(b, host.data(), host.size() * 4);
    return b;
  }

  // Runs one kernel. Empty string = P matches the CPU reference and nothing
  // outside P was written.
  std::string run(Mode mode) {
    const size_t e = (size_t)w * w;
    std::vector<float> host(e + slack, kCanary);
    std::fill(host.begin(), host.begin() + e, kUnwritten);
    pmpp::upload(P, host.data(), host.size() * 4);
    runMode(mode, M, N, P, w);
    pmpp::download(P, host.data(), host.size() * 4);
    std::string msg = pmpp::describe(pmpp::compare(host.data(), ref.data(), mag.data(), e), w);
    size_t unwritten = 0, past = 0;
    for (size_t i = 0; i < e; ++i) unwritten += (host[i] == kUnwritten);
    for (size_t i = e; i < host.size(); ++i) past += !(host[i] == kCanary);
    auto add = [&](const std::string& s) { msg += (msg.empty() ? "" : "; ") + s; };
    if (unwritten) add(std::to_string(unwritten) + " of " + std::to_string(e) + " elements of P were never written");
    if (past) add(std::to_string(past) + " values were stored past the end of P");
    return msg;
  }
};

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  int ran[3] = {0, 0, 0}, ok[3] = {0, 0, 0};
  // 1 is a single thread's worth; 4 is the book's Fig. 3.12; 16 and 33 sit on and
  // just past the 16 x 16 blocks of the provided kernel; 100 and 257 leave most or
  // all but one of the threads of a 256-thread block without work; 512 is two whole blocks.
  std::vector<int> widths = {1, 4, 16, 33, 100};
  if (!kEmulated) { widths.push_back(257); widths.push_back(512); }
  for (int mi = 0; mi < 3; ++mi) {
    const Mode mode = (Mode)mi;
    const std::string shapeProblem = launchShapeProblem(mode);
    if (!shapeProblem.empty()) {
      t.check(std::string(kModeLabel[mi]) + ": launch usable here", [&] { ran[mi]++; return shapeProblem; });
      continue;
    }
    for (int w : widths) {
      t.check(std::string(kModeLabel[mi]) + ", Width " + std::to_string(w), [&]() -> std::string {
        Guarded g(w, 500 + w);
        std::string err = g.run(mode);
        ran[mi]++; ok[mi] += err.empty();
        return err;
      });
    }
  }
  std::printf("per kernel: element %d/%d, row %d/%d, col %d/%d\n", ok[0], ran[0], ok[1], ran[1], ok[2], ran[2]);
  return t.finish();
}

static int bench(const pmpp::Args& a) {
  // The three kernels do the same 2 * Width^3 floating-point operations. Choose one
  // width for all of them: double it until the per-element kernel runs long enough
  // to time, but stop early if another doubling (8 times the arithmetic) would push
  // the slowest kernel past a budget of half a second per run.
  const double longEnough = 0.010, budget = 0.5;
  const int maxWidth = 4096;
  int w = 256;
  const char* stopped = "";
  for (;;) {
    pmpp::GemmProblem p(w, w, w, 11, false);
    double s[3];
    for (int mi = 0; mi < 3; ++mi) s[mi] = pmpp::time_it(1, [&] { return runMode((Mode)mi, p.A, p.B, p.C, w); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s[0] <= 0 || s[1] <= 0 || s[2] <= 0) { std::printf("failed at width %d: %s\n", w, err.c_str()); return 1; }
    const double slowest = std::max(s[0], std::max(s[1], s[2]));
    if (s[0] >= longEnough) { stopped = "the per-element kernel is long enough to time"; break; }
    if (slowest * 8 > budget) { stopped = "one more doubling would make the slowest kernel too slow"; break; }
    if (w >= maxWidth) { stopped = "the driver's largest width"; break; }
    w *= 2;
  }

  pmpp::GemmProblem p(w, w, w, 11, false);
  const double flop = p.flop();
  std::printf("Width %d (%.2f GFLOP per multiplication; chosen because %s), best of %d runs\n", w, flop / 1e9, stopped, a.reps);
  std::printf("  %-34s %10s %12s %12s %14s\n", "kernel", "threads", "time (ms)", "GFLOP/s", "time / element");
  double element_s = 0;
  for (int mi = 0; mi < 3; ++mi) {
    const double s = pmpp::time_it(a.reps, [&] { return runMode((Mode)mi, p.A, p.B, p.C, w); }).min_s;
    std::string err = pmpp::take_error();
    if (!err.empty() || s <= 0) { std::printf("  %s failed: %s\n", kModeLabel[mi], err.c_str()); return 1; }
    if (mi == kElement) element_s = s;
    // Threads that have work, and threads the rounded-up grid launches.
    const double threads = mi == kElement ? (double)w * w : (double)w;
    const double perBlock = mi == kElement ? 16 : 256;
    const double blocks = std::ceil(w / perBlock);
    const double launched = mi == kElement ? blocks * blocks * 256 : blocks * 256;
    std::printf("  %-34s %10.0f %12.4f %12.2f %14.2f\n", kModeLabel[mi], threads, s * 1e3, flop / s / 1e9, s / element_s);
    pmpp::bench_row(a, "granularity").s("mode", kModeName[mi]).n("width", w).n("threads", threads)
        .n("launched", launched).n("seconds", s).n("gflops", flop / s / 1e9).n("time_vs_element", s / element_s).emit();
  }
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
