// Lab 6.1 driver: tests your three access-pattern kernels against the CPU, then
// measures what each pattern costs on this device next to a provided copy.
// You do not need to edit this file; your work goes in work/.
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- entry points implemented in <variant>/ and provided/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeStridedRead(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                       id<MTLBuffer> output, int32_t n, int32_t stride);
void encodeTransposeReadRows(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                             id<MTLBuffer> output, int32_t rows, int32_t cols);
void encodeTransposeWriteRows(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                              id<MTLBuffer> output, int32_t rows, int32_t cols);
void encodeCopy(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> input,
                id<MTLBuffer> output, int32_t n);
using pmpp::metal::buffer;

// provided/copy.metal, found from this variant's kernel.metal: <lab>/<variant>/kernel.metal.
static std::string providedMetal() {
  std::string s = PMPP_METAL_SOURCE;
  s = s.substr(0, s.rfind('/'));
  s = s.substr(0, s.rfind('/'));
  return s + "/provided/copy.metal";
}
static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
static id<MTLComputePipelineState> copyPso() { return pmpp::metal::pipeline(providedMetal(), "copy_floats"); }
static pmpp::Timing runStrided(Buf* in, Buf* out, int n, int stride) {
  auto p = pso("strided_read"); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeStridedRead(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), n, stride); });
}
static pmpp::Timing runReadRows(Buf* in, Buf* out, int rows, int cols) {
  auto p = pso("transpose_read_rows"); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeTransposeReadRows(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), rows, cols); });
}
static pmpp::Timing runWriteRows(Buf* in, Buf* out, int rows, int cols) {
  auto p = pso("transpose_write_rows"); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeTransposeWriteRows(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), rows, cols); });
}
static pmpp::Timing runCopy(Buf* in, Buf* out, int n) {
  auto p = copyPso(); if (!p) return {0, 0};
  return pmpp::run([&](void* v) { encodeCopy(((pmpp::metal::Pass*)v)->encoder, p, buffer(in), buffer(out), n); });
}
// The SIMD-group width, read from a pipeline as Apple documents (never assumed).
static int simdWidth() { auto p = copyPso(); return p ? (int)p.threadExecutionWidth : 0; }
static const char* simdWidthSource() { return "threadExecutionWidth of the copy pipeline"; }
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static double maxBuffer() { return pmpp::device_prop("maxBufferLength", 0); }
#else
void launchStridedRead(const float* in, float* out, int n, int stride);
void launchTransposeReadRows(const float* in, float* out, int rows, int cols);
void launchTransposeWriteRows(const float* in, float* out, int rows, int cols);
void launchCopy(const float* in, float* out, int n);
static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static pmpp::Timing runStrided(Buf* in, Buf* out, int n, int stride) {
  return pmpp::run([&](void*) { launchStridedRead(f(in), f(out), n, stride); });
}
static pmpp::Timing runReadRows(Buf* in, Buf* out, int rows, int cols) {
  return pmpp::run([&](void*) { launchTransposeReadRows(f(in), f(out), rows, cols); });
}
static pmpp::Timing runWriteRows(Buf* in, Buf* out, int rows, int cols) {
  return pmpp::run([&](void*) { launchTransposeWriteRows(f(in), f(out), rows, cols); });
}
static pmpp::Timing runCopy(Buf* in, Buf* out, int n) {
  return pmpp::run([&](void*) { launchCopy(f(in), f(out), n); });
}
static int simdWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static const char* simdWidthSource() { return "warpSize from the device properties"; }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static double maxBuffer() { return 0; }
#endif

// ---------------------------------------------------------------- helpers
// Fills a device buffer with NaN, so an element the kernel never writes fails the
// exact comparison (NaN equals nothing, not even itself).
static void poison(Buf* b, size_t n) {
  std::vector<float> v(n, std::numeric_limits<float>::quiet_NaN());
  pmpp::upload(b, v.data(), n * 4);
}

// Copies must be exact: no arithmetic happens, so any difference is a wrong index.
static std::string exact(const std::vector<float>& got, const std::vector<float>& ref, size_t cols) {
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    if (!(got[i] == ref[i])) { if (bad == 0) first = i; ++bad; }
  }
  if (bad == 0) return "";
  char b[240];
  std::snprintf(b, sizeof b, "%zu of %zu elements wrong; first at row %zu, col %zu: got %g%s, expected %g",
                bad, got.size(), first / cols, first % cols, (double)got[first],
                std::isnan(got[first]) ? " (never written)" : "", (double)ref[first]);
  return b;
}

// ---------------------------------------------------------------- tests
static std::string testStrided(int n, int stride) {
  const size_t inN = (size_t)n * stride;   // exactly what the kernel may read, nothing more
  auto x = pmpp::random_floats(inN, 600 + n + 7 * stride);
  Buf* I = pmpp::alloc(inN * 4);
  Buf* O = pmpp::alloc((size_t)n * 4);
  pmpp::upload(I, x.data(), inN * 4);
  poison(O, n);
  runStrided(I, O, n, stride);
  std::vector<float> got(n), ref(n);
  pmpp::download(O, got.data(), (size_t)n * 4);
  for (int i = 0; i < n; ++i) ref[i] = x[(size_t)i * stride];
  pmpp::release(I); pmpp::release(O);
  return exact(got, ref, (size_t)n);
}

static std::string testTranspose(bool readRows, int rows, int cols) {
  const size_t e = (size_t)rows * cols;
  auto x = pmpp::random_floats(e, 700 + rows * 3 + cols);
  Buf* I = pmpp::alloc(e * 4);
  Buf* O = pmpp::alloc(e * 4);
  pmpp::upload(I, x.data(), e * 4);
  poison(O, e);
  if (readRows) runReadRows(I, O, rows, cols); else runWriteRows(I, O, rows, cols);
  std::vector<float> got(e), ref(e);
  pmpp::download(O, got.data(), e * 4);
  for (int r = 0; r < rows; ++r)
    for (int c = 0; c < cols; ++c) ref[(size_t)c * rows + r] = x[(size_t)r * cols + c];
  pmpp::release(I); pmpp::release(O);
  return exact(got, ref, (size_t)rows);   // out is cols x rows
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  struct S { int n, stride; };
  std::vector<S> strided;
  for (int n : {1, 31, 32, 33, 1000})
    for (int s : {1, 2, 3, 8, 32}) strided.push_back({n, s});
  if (!pmpp::is_emu()) strided.push_back({65536, 64});
  for (const S& s : strided) {
    char name[120];
    std::snprintf(name, sizeof name, "strided read, n = %d, stride = %d", s.n, s.stride);
    t.check(name, [&] { return testStrided(s.n, s.stride); });
  }

  struct Shape { int rows, cols; const char* why; };
  std::vector<Shape> shapes = {
      {1, 1, "one element"},
      {1, 37, "a single row"},
      {37, 1, "a single column"},
      {32, 32, "two whole blocks each way"},
      {33, 65, "partial blocks in both directions"},
      {100, 7, "tall and narrow"},
  };
  if (!pmpp::is_emu()) {
    shapes.push_back({1000, 513, "large and awkward"});
    shapes.push_back({2048, 2048, "large and square"});
  }
  for (const Shape& s : shapes) {
    for (bool readRows : {true, false}) {
      char name[160];
      std::snprintf(name, sizeof name, "transpose, rows %s, %d x %d  (%s)",
                    readRows ? "read" : "written", s.rows, s.cols, s.why);
      t.check(name, [&] { return testTranspose(readRows, s.rows, s.cols); });
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// One benchmark row. Rows that time one of several kernels name it in "variant"
// (the field the lesson pages read) and again in "kernel": the verify report's
// tables leave "variant" out, because there it means work or solution, and the
// result that holds these rows already says which of the two ran.
static pmpp::Json row(const pmpp::Args& a, const char* name, const char* kernel = nullptr) {
  pmpp::Json j;
  j.s("kind", "bench").s("lab", a.lab).s("backend", pmpp::backend()).s("name", name);
  if (kernel) j.s("variant", kernel).s("kernel", kernel);
  else j.s("variant", a.variant);
  return j;
}

static bool failed(const std::vector<double>& s, const char* what) {
  std::string err = pmpp::take_error();
  if (err.empty() && *std::min_element(s.begin(), s.end()) > 0) return false;
  std::printf("  %s failed to run: %s\n", what, err.empty() ? "no time reported" : err.c_str());
  return true;
}

// Best time of each kernel over `reps` rounds, after one warm-up of each. The
// kernels take turns within a round, so a change of clock speed, or other work on
// the machine, during the measurement affects them alike.
static std::vector<double> bestOfRounds(int reps, const std::vector<std::function<pmpp::Timing()>>& runs) {
  std::vector<double> best(runs.size(), 0);
  for (auto& r : runs) r();
  for (int i = 0; i < reps; ++i) {
    for (size_t k = 0; k < runs.size(); ++k) {
      const double s = runs[k]().best();
      if (i == 0 || s < best[k]) best[k] = s;
    }
  }
  return best;
}

static int bench(const pmpp::Args& a) {
  // A budget for the buffers, from what the device reports: a quarter of its memory,
  // and no single buffer above the device's buffer limit where it has one.
  const double mem = deviceMemory(), maxBuf = maxBuffer();
  auto fits = [&](double bufferBytes, int buffers) {
    return (mem <= 0 || buffers * bufferBytes <= mem / 4) && (maxBuf <= 0 || bufferBytes <= maxBuf);
  };

  // ---- copy and strided read over one fixed input of N floats ----
  int N = 1 << 25;
  while (N > (1 << 20) && !fits(4.0 * N, 2)) N /= 2;
  Buf* I = pmpp::alloc((size_t)N * 4);
  Buf* O = pmpp::alloc((size_t)N * 4);
  std::vector<float> zeros((size_t)N, 0.0f);
  pmpp::upload(I, zeros.data(), zeros.size() * 4);
  std::printf("input of %d floats (%.0f MiB); best of %d runs after a warm-up, the kernels of each\n"
              "group below taking turns\n", N, N * 4.0 / (1 << 20), a.reps);

  const std::vector<int> strides = {1, 2, 4, 8, 16, 32, 64};
  std::vector<std::function<pmpp::Timing()>> runs = {[&] { return runCopy(I, O, N); }};
  for (int stride : strides) runs.push_back([&, stride] { return runStrided(I, O, N / stride, stride); });
  std::vector<double> s = bestOfRounds(a.reps, runs);
  if (failed(s, "copy or strided read")) return 1;

  const double copyBw = 8.0 * N / s[0] / 1e9;
  std::printf("\n  copy (provided)  n = %d  %9.4f ms  %8.1f GB/s\n", N, s[0] * 1e3, copyBw);
  row(a, "copy").n("n", N).n("seconds", s[0]).n("gb_per_s", copyBw).emit();

  std::printf("\n  strided read: out[i] = in[i * stride], n = %d / stride\n", N);
  std::printf("  GB/s counts the bytes the kernel uses: 4 read and 4 written per element\n");
  std::printf("  %6s %10s %12s %10s %14s\n", "stride", "n", "time (ms)", "GB/s", "vs stride 1");
  const double bw1 = 8.0 * N / s[1] / 1e9;
  for (size_t k = 0; k < strides.size(); ++k) {
    const int stride = strides[k], n = N / stride;
    const double t = s[k + 1], bw = 8.0 * n / t / 1e9;
    std::printf("  %6d %10d %12.4f %10.1f %13.1f%%\n", stride, n, t * 1e3, bw, 100.0 * bw / bw1);
    row(a, "strided").n("stride", stride).n("n", n).n("seconds", t).n("gb_per_s", bw)
        .n("relative_to_stride1", bw / bw1).emit();
  }
  pmpp::release(I); pmpp::release(O);

  // ---- the two naive transposes ----
  int side = 4096;
  while (side > 256 && !fits(4.0 * side * side, 2)) side /= 2;
  I = pmpp::alloc((size_t)side * side * 4);
  O = pmpp::alloc((size_t)side * side * 4);
  zeros.assign((size_t)side * side, 0.0f);
  pmpp::upload(I, zeros.data(), zeros.size() * 4);
  std::printf("\n  transpose of a %d x %d matrix (GB/s counts 4 bytes read and 4 written per element)\n", side, side);
  s = bestOfRounds(a.reps, {[&] { return runReadRows(I, O, side, side); },
                            [&] { return runWriteRows(I, O, side, side); }});
  if (failed(s, "transpose")) return 1;
  for (bool readRows : {true, false}) {
    const double t = s[readRows ? 0 : 1], bw = 8.0 * side * (double)side / t / 1e9;
    std::printf("  %-38s %9.4f ms  %8.1f GB/s\n",
                readRows ? "rows read (strided writes)" : "rows written (strided reads)", t * 1e3, bw);
    row(a, "transpose", readRows ? "read_rows" : "write_rows").n("rows", side).n("cols", side)
        .n("seconds", t).n("gb_per_s", bw).emit();
  }
  pmpp::release(I); pmpp::release(O);

  // ---- the width this device executes together ----
  const int w = simdWidth();
  std::printf("\n  SIMD width: %d (%s)\n", w, simdWidthSource());
  if (w > 0) row(a, "simd_width").n("width", w).emit();
  std::printf("  These are measurements of this machine, not expectations.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
