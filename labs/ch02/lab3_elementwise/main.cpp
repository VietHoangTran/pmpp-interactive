// Lab 2.3 driver: tests and benchmarks the three elementwise kernels of project
// milestone M0, and the function that chains them with one round trip.
// You do not need to edit this file; your work goes in work/.
//
// The kernels are tested the way later milestones will use them: on tensors the
// caller (here, the driver) already keeps on the device. blockTail is tested
// the way main() in the book's Fig. 2.4 would use it: host arrays in, a host
// array out.
#include <chrono>
#include <limits>

#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---- the CPU reference (project convention: it lives in main.cpp) ----
// Tensors are row-major, rows first: element (r, j) is at r * d + j.
static void cpuAddBias(float* Y, const float* X, const float* bias, int rows, int d) {
  for (int r = 0; r < rows; ++r)
    for (int j = 0; j < d; ++j) Y[(size_t)r * d + j] = X[(size_t)r * d + j] + bias[j];
}
static void cpuScaleShift(float* Y, const float* X, float alpha, float beta, size_t n) {
  for (size_t i = 0; i < n; ++i) Y[i] = alpha * X[i] + beta;
}
static void cpuAddResidual(float* X, const float* R, size_t n) {
  for (size_t i = 0; i < n; ++i) X[i] = X[i] + R[i];
}
// y = alpha * (x + bias) + beta + r
static void cpuBlockTail(const float* x, const float* bias, const float* r, float alpha, float beta,
                         float* y, int rows, int d) {
  const size_t n = (size_t)rows * d;
  cpuAddBias(y, x, bias, rows, d);
  cpuScaleShift(y, y, alpha, beta, n);
  cpuAddResidual(y, r, n);
}

static double seconds(std::chrono::steady_clock::time_point from, std::chrono::steady_clock::time_point to) {
  return std::chrono::duration<double>(to - from).count();
}

// ---- entry points implemented in <variant>/ ----
#if defined(PMPP_BACKEND_METAL)
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeAddBias(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                   id<MTLBuffer> Y, id<MTLBuffer> X, id<MTLBuffer> bias, uint32_t rows, uint32_t d);
void encodeScaleShift(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                      id<MTLBuffer> Y, id<MTLBuffer> X, float alpha, float beta, uint32_t n);
void encodeAddResidual(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> pipeline,
                       id<MTLBuffer> X, id<MTLBuffer> R, uint32_t n);
int blockTail(id<MTLDevice> device, id<MTLCommandQueue> queue,
              id<MTLComputePipelineState> addBias, id<MTLComputePipelineState> scaleShift,
              id<MTLComputePipelineState> addResidual,
              const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
              float* y_h, uint32_t rows, uint32_t d);

using pmpp::metal::buffer;
static id<MTLComputePipelineState> pso(const char* fn) { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, fn); }
static id<MTLCommandQueue> queue() {
  static id<MTLCommandQueue> q = [pmpp::metal::device() newCommandQueue];
  return q;
}
// Empty when kernel.metal compiled and holds the three kernels.
static std::string setupError() {
  for (const char* fn : {"add_bias", "scale_shift_k", "add_residual"}) {
    if (pso(fn)) continue;
    std::string e = pmpp::take_error();
    return e.empty() ? std::string("no pipeline for kernel ") + fn : e;
  }
  return "";
}
static pmpp::Timing runAddBias(Buf* Y, Buf* X, Buf* bias, int rows, int d) {
  return pmpp::run([&](void* v) {
    encodeAddBias(((pmpp::metal::Pass*)v)->encoder, pso("add_bias"), buffer(Y), buffer(X), buffer(bias),
                  (uint32_t)rows, (uint32_t)d);
  });
}
static pmpp::Timing runScaleShift(Buf* Y, Buf* X, float alpha, float beta, size_t n) {
  return pmpp::run([&](void* v) {
    encodeScaleShift(((pmpp::metal::Pass*)v)->encoder, pso("scale_shift_k"), buffer(Y), buffer(X), alpha, beta, (uint32_t)n);
  });
}
static pmpp::Timing runAddResidual(Buf* X, Buf* R, size_t n) {
  return pmpp::run([&](void* v) {
    encodeAddResidual(((pmpp::metal::Pass*)v)->encoder, pso("add_residual"), buffer(X), buffer(R), (uint32_t)n);
  });
}
static int callBlockTail(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
                         float* y_h, int rows, int d) {
  @autoreleasepool {
    return blockTail(pmpp::metal::device(), queue(), pso("add_bias"), pso("scale_shift_k"), pso("add_residual"),
                     x_h, bias_h, r_h, alpha, beta, y_h, (uint32_t)rows, (uint32_t)d);
  }
}

// The benchmark's other way of computing the chain: three separate offloads
// built from your encode functions, each with its own buffers, its own command
// buffer and its own wait, and host arrays in between. On Metal "copy in" means
// creating a shared buffer from a host array.
static bool submit(const std::function<void(id<MTLComputeCommandEncoder>)>& encode) {
  id<MTLCommandBuffer> commandBuffer = [queue() commandBuffer];
  id<MTLComputeCommandEncoder> encoder = [commandBuffer computeCommandEncoder];
  if (!commandBuffer || !encoder) return false;
  encode(encoder);
  [encoder endEncoding];
  [commandBuffer commit];
  [commandBuffer waitUntilCompleted];
  return commandBuffer.status == MTLCommandBufferStatusCompleted;
}
static int threeRoundTrips(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
                           float* y_h, int rows, int d) {
  id<MTLDevice> dev = pmpp::metal::device();
  const uint32_t n = (uint32_t)rows * (uint32_t)d;
  const NSUInteger bytes = (NSUInteger)n * sizeof(float);
  const MTLResourceOptions shared = MTLResourceStorageModeShared;
  @autoreleasepool {   // offload 1: y = x + bias
    id<MTLBuffer> X = [dev newBufferWithBytes:x_h length:bytes options:shared];
    id<MTLBuffer> B = [dev newBufferWithBytes:bias_h length:(NSUInteger)d * sizeof(float) options:shared];
    id<MTLBuffer> Y = [dev newBufferWithLength:bytes options:shared];
    if (!X || !B || !Y) return 1;
    if (!submit([&](id<MTLComputeCommandEncoder> enc) {
          encodeAddBias(enc, pso("add_bias"), Y, X, B, (uint32_t)rows, (uint32_t)d);
        })) return 1;
    std::memcpy(y_h, Y.contents, bytes);
  }
  @autoreleasepool {   // offload 2: y = alpha * y + beta
    id<MTLBuffer> X = [dev newBufferWithBytes:y_h length:bytes options:shared];
    id<MTLBuffer> Y = [dev newBufferWithLength:bytes options:shared];
    if (!X || !Y) return 1;
    if (!submit([&](id<MTLComputeCommandEncoder> enc) {
          encodeScaleShift(enc, pso("scale_shift_k"), Y, X, alpha, beta, n);
        })) return 1;
    std::memcpy(y_h, Y.contents, bytes);
  }
  @autoreleasepool {   // offload 3: y = y + r
    id<MTLBuffer> X = [dev newBufferWithBytes:y_h length:bytes options:shared];
    id<MTLBuffer> R = [dev newBufferWithBytes:r_h length:bytes options:shared];
    if (!X || !R) return 1;
    if (!submit([&](id<MTLComputeCommandEncoder> enc) {
          encodeAddResidual(enc, pso("add_residual"), X, R, n);
        })) return 1;
    std::memcpy(y_h, X.contents, bytes);
  }
  return 0;
}
static const char* roundTripKind() { return "host arrays copied into shared buffers, results copied out"; }
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static double largestBuffer() { return pmpp::device_prop("maxBufferLength", 0); }
#else
float scaleShiftOnHost(float x, float alpha, float beta);
void launchAddBias(float* Y_d, const float* X_d, const float* bias_d, int rows, int d);
void launchScaleShift(float* Y_d, const float* X_d, float alpha, float beta, int n);
void launchAddResidual(float* X_d, const float* R_d, int n);
int blockTail(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
              float* y_h, int rows, int d);
#define PMPP_LAB_HAS_HOST_HELPER 1

static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static std::string setupError() { return ""; }
static pmpp::Timing runAddBias(Buf* Y, Buf* X, Buf* bias, int rows, int d) {
  return pmpp::run([&](void*) { launchAddBias(f(Y), f(X), f(bias), rows, d); });
}
static pmpp::Timing runScaleShift(Buf* Y, Buf* X, float alpha, float beta, size_t n) {
  return pmpp::run([&](void*) { launchScaleShift(f(Y), f(X), alpha, beta, (int)n); });
}
static pmpp::Timing runAddResidual(Buf* X, Buf* R, size_t n) {
  return pmpp::run([&](void*) { launchAddResidual(f(X), f(R), (int)n); });
}
static int callBlockTail(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
                         float* y_h, int rows, int d) {
  return blockTail(x_h, bias_h, r_h, alpha, beta, y_h, rows, d);
}

// The benchmark's other way of computing the chain: three separate offloads
// built from your launch functions, each shaped like the book's Fig. 2.13
// (allocate, copy in, launch, wait, copy out, free), with host arrays in between.
static int threeRoundTrips(const float* x_h, const float* bias_h, const float* r_h, float alpha, float beta,
                           float* y_h, int rows, int d) {
  const size_t n = (size_t)rows * d, bytes = n * sizeof(float), biasBytes = (size_t)d * sizeof(float);
  auto clean = [] { return pmpp::take_error().empty(); };
  bool ok;
  {   // offload 1: y = x + bias
    Buf *X = pmpp::alloc(bytes), *B = pmpp::alloc(biasBytes), *Y = pmpp::alloc(bytes);
    if ((ok = clean())) {
      pmpp::upload(X, x_h, bytes);
      pmpp::upload(B, bias_h, biasBytes);
      runAddBias(Y, X, B, rows, d);
      pmpp::download(Y, y_h, bytes);
      ok = clean();
    }
    pmpp::release(X); pmpp::release(B); pmpp::release(Y);
    if (!ok) return 1;
  }
  {   // offload 2: y = alpha * y + beta
    Buf *X = pmpp::alloc(bytes), *Y = pmpp::alloc(bytes);
    if ((ok = clean())) {
      pmpp::upload(X, y_h, bytes);
      runScaleShift(Y, X, alpha, beta, n);
      pmpp::download(Y, y_h, bytes);
      ok = clean();
    }
    pmpp::release(X); pmpp::release(Y);
    if (!ok) return 1;
  }
  {   // offload 3: y = y + r
    Buf *X = pmpp::alloc(bytes), *R = pmpp::alloc(bytes);
    if ((ok = clean())) {
      pmpp::upload(X, y_h, bytes);
      pmpp::upload(R, r_h, bytes);
      runAddResidual(X, R, n);
      pmpp::download(X, y_h, bytes);
      ok = clean();
    }
    pmpp::release(X); pmpp::release(R);
    if (!ok) return 1;
  }
  return 0;
}
static const char* roundTripKind() { return "allocate, copy in, launch, wait, copy out, free"; }
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static double largestBuffer() { return 0; }
#endif

// ---- test data ----
struct Shape { int rows, d; };
static std::string shapeName(const Shape& s) { return std::to_string(s.rows) + " x " + std::to_string(s.d); }

// Every value is well away from zero, and so is every correct result: an
// element no thread wrote cannot look right, whatever the device memory held.
// The bias differs from column to column, and apart from 1 x 1 every tested
// shape has fewer rows than columns, so a bias indexed by row reads the wrong
// values and a mixed-up row and column lands on the wrong element.
struct Data {
  int rows, d;
  size_t n;
  std::vector<float> x, bias, r;
  Data(const Shape& s, uint32_t seed) : rows(s.rows), d(s.d), n((size_t)s.rows * s.d) {
    x = pmpp::random_floats(n, seed);
    bias = pmpp::random_floats((size_t)d, seed + 1);
    r = pmpp::random_floats(n, seed + 2);
    for (float& v : x) v = 1.5f + 0.5f * v;      // [1, 2]
    for (float& v : bias) v = 3.0f + v;          // [2, 4]
    for (float& v : r) v = 1.5f + 0.5f * v;      // [1, 2]
  }
};
static const float kAlpha = 0.5f, kBeta = 2.0f;

// On a GPU a thread that runs past the end of a tensor can land in memory that
// exists, and then nothing reports it (ch02-facts A6, C5). So on real devices
// the driver allocates 1024 extra elements behind every tensor (more than any
// block or threadgroup can overshoot by), fills them with a known value and
// looks at them afterwards. The emulator gets tensors of the exact size
// instead: AddressSanitizer then stops at the access itself and names the
// kernel line.
static size_t paddingElements() { return pmpp::is_emu() ? 0 : 1024; }
static const float kBehindTheEnd = 123456.0f;

// A tensor the driver keeps on the device.
struct Dev {
  size_t n;                  // elements of the tensor itself
  std::vector<float> up;     // what was uploaded: the tensor, then the padding
  Buf* buf;
  explicit Dev(const std::vector<float>& v) : n(v.size()), up(v) {
    up.resize(n + paddingElements(), kBehindTheEnd);
    buf = pmpp::alloc(up.size() * sizeof(float));
    pmpp::upload(buf, up.data(), up.size() * sizeof(float));
  }
  Dev(const Dev&) = delete;
  ~Dev() { pmpp::release(buf); }
  std::vector<float> fetch() const {
    std::vector<float> got(up.size());
    pmpp::download(buf, got.data(), got.size() * sizeof(float));
    return got;
  }
  // Empty if the device still holds exactly what the driver uploaded.
  std::string unchanged(const char* what) const {
    const std::vector<float> got = fetch();
    if (std::memcmp(got.data(), up.data(), up.size() * sizeof(float)) == 0) return "";
    return std::string(what) + " was modified: it is an input and must only be read";
  }
  // Empty if the tensor equals `ref` within the tolerance and nothing was
  // written behind it.
  std::string equals(const std::vector<float>& ref, int d, const char* what) const {
    const std::vector<float> got = fetch();
    const pmpp::Cmp c = pmpp::compare(got.data(), ref.data(), nullptr, n);
    if (!c.ok) {
      const size_t i = c.first_bad;
      char m[320];
      std::snprintf(m, sizeof m, "%zu of %zu elements of %s wrong; first at row %zu, column %zu: got %g, expected %g%s",
                    c.bad, n, what, i / (size_t)d, i % (size_t)d, (double)got[i], (double)ref[i],
                    std::isnan(got[i]) ? " (still the NaN the driver put there: no thread wrote it)" : "");
      return m;
    }
    size_t over = 0, first = 0;
    for (size_t i = n; i < got.size(); ++i)
      if (std::memcmp(&got[i], &up[i], sizeof(float)) != 0) { if (over++ == 0) first = i; }
    if (over == 0) return "";
    char m[240];
    std::snprintf(m, sizeof m, "%s is correct, but %zu elements behind its end were overwritten (first at flat index %zu; "
                  "the tensor ends at %zu): threads past the end must do nothing", what, over, first, n);
    return m;
  }
};
static std::vector<float> nans(size_t n) { return std::vector<float>(n, std::numeric_limits<float>::quiet_NaN()); }

// ---- the tests ----
static std::string testAddBias(const Shape& s) {
  Data D(s, 100u + (uint32_t)s.d);
  Dev X(D.x), B(D.bias), Y(nans(D.n));       // Y starts as NaN
  runAddBias(Y.buf, X.buf, B.buf, s.rows, s.d);
  std::vector<float> ref(D.n);
  cpuAddBias(ref.data(), D.x.data(), D.bias.data(), s.rows, s.d);
  std::string e = Y.equals(ref, s.d, "Y");
  if (e.empty()) e = X.unchanged("X");
  if (e.empty()) e = B.unchanged("bias");
  return e;
}

static std::string testScaleShift(const Shape& s, bool againstHostHelper) {
  Data D(s, 200u + (uint32_t)s.d);
  Dev X(D.x), Y(nans(D.n));
  runScaleShift(Y.buf, X.buf, kAlpha, kBeta, D.n);
  std::vector<float> ref(D.n);
  if (againstHostHelper) {
#if defined(PMPP_LAB_HAS_HOST_HELPER)
    // The host compilation of your scaleShift, element by element. Not compared
    // bit for bit: a compiler may turn alpha * x + beta into one fused
    // multiply-add, which rounds once instead of twice, and the two
    // compilations need not make the same choice. (Whether nvcc or the HIP
    // compiler does so by default is not something we have checked.)
    for (size_t i = 0; i < D.n; ++i) ref[i] = scaleShiftOnHost(D.x[i], kAlpha, kBeta);
#endif
  } else {
    cpuScaleShift(ref.data(), D.x.data(), kAlpha, kBeta, D.n);
  }
  std::string e = Y.equals(ref, s.d, againstHostHelper ? "Y (compared with scaleShiftOnHost)" : "Y");
  if (e.empty()) e = X.unchanged("X");
  return e;
}

static std::string testAddResidual(const Shape& s) {
  Data D(s, 300u + (uint32_t)s.d);
  Dev X(D.x), R(D.r);
  runAddResidual(X.buf, R.buf, D.n);
  std::vector<float> ref = D.x;
  cpuAddResidual(ref.data(), D.r.data(), D.n);
  std::string e = X.equals(ref, s.d, "X");
  if (e.empty()) e = R.unchanged("R");
  return e;
}

static std::string wrongHostResult(const std::vector<float>& y, const std::vector<float>& ref, int d) {
  const pmpp::Cmp c = pmpp::compare(y.data(), ref.data(), nullptr, y.size());
  if (c.ok) return "";
  size_t stillNaN = 0;
  for (float v : y) stillNaN += std::isnan(v) ? 1 : 0;
  const size_t i = c.first_bad;
  char m[360];
  int len = std::snprintf(m, sizeof m, "%zu of %zu elements of y_h wrong; first at row %zu, column %zu: got %g, expected %g",
                          c.bad, y.size(), i / (size_t)d, i % (size_t)d, (double)y[i], (double)ref[i]);
  if (stillNaN > 0 && len > 0 && (size_t)len < sizeof m)
    std::snprintf(m + len, sizeof m - (size_t)len, "; %zu elements still hold the NaN the driver put there, so nothing was copied back to them",
                  stillNaN);
  return m;
}

static std::string testBlockTail(const Shape& s) {
  Data D(s, 400u + (uint32_t)s.d);
  const std::vector<float> x0 = D.x, bias0 = D.bias, r0 = D.r;
  std::vector<float> y = nans(D.n), ref(D.n);
  cpuBlockTail(D.x.data(), D.bias.data(), D.r.data(), kAlpha, kBeta, ref.data(), s.rows, s.d);
  const int rc = callBlockTail(D.x.data(), D.bias.data(), D.r.data(), kAlpha, kBeta, y.data(), s.rows, s.d);
  if (rc != 0) return "blockTail returned " + std::to_string(rc) + "; a call that works must return 0";
  std::string e = wrongHostResult(y, ref, s.d);
  if (!e.empty()) return e;
  if (D.x != x0) return "x_h was modified: the inputs are read-only";
  if (D.bias != bias0) return "bias_h was modified: the inputs are read-only";
  if (D.r != r0) return "r_h was modified: the inputs are read-only";
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const std::string broken = setupError();
  if (!broken.empty()) {
    t.check("kernel.metal builds into three pipelines", [&] { return broken; });
    return t.finish();
  }

  // rows * d is 1, 257, 300, 1024 and 231: one multiple of the block size, four
  // that are not. Real devices add a tensor of 64 x 4099.
  std::vector<Shape> shapes = {{1, 1}, {1, 257}, {3, 100}, {4, 256}, {7, 33}};
  if (!pmpp::is_emu()) shapes.push_back({64, 4099});

  for (const Shape& s : shapes)
    t.check("addBias, " + shapeName(s), [&] { return testAddBias(s); });

#if defined(PMPP_LAB_HAS_HOST_HELPER)
  t.check("scaleShiftOnHost(x, alpha, beta) is alpha * x + beta", []() -> std::string {
    for (float x : {1.0f, 1.25f, 2.0f, -3.0f}) {
      const float got = scaleShiftOnHost(x, kAlpha, kBeta), want = kAlpha * x + kBeta;
      if (!(std::fabs(got - want) <= 1e-5f)) {
        char m[160];
        std::snprintf(m, sizeof m, "scaleShiftOnHost(%g, %g, %g) returned %g, expected %g", (double)x, (double)kAlpha,
                      (double)kBeta, (double)got, (double)want);
        return m;
      }
    }
    return "";
  });
#endif
  for (const Shape& s : shapes)
    t.check("scaleShift, " + shapeName(s), [&] { return testScaleShift(s, false); });
#if defined(PMPP_LAB_HAS_HOST_HELPER)
  for (const Shape& s : shapes)
    t.check("scaleShift kernel against scaleShiftOnHost, " + shapeName(s), [&] { return testScaleShift(s, true); });
#endif

  for (const Shape& s : shapes)
    t.check("addResidual in place, " + shapeName(s), [&] { return testAddResidual(s); });

  for (const Shape& s : shapes)
    t.check("blockTail, " + shapeName(s), [&] { return testBlockTail(s); });
  return t.finish();
}

// ---- the benchmark (real devices only) ----
template <class F>
static double bestWall(int reps, F&& once) {
  double best = 1e300;
  for (int r = 0; r < reps; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    once();
    best = std::min(best, seconds(t0, std::chrono::steady_clock::now()));
  }
  return best;
}

static int bench(const pmpp::Args& a) {
  const std::string broken = setupError();
  if (!broken.empty()) { std::printf("bench skipped: %s\n", broken.c_str()); return 1; }

  const int d = 4096;
  const size_t maxElements = (size_t)1 << 26;
  const double mem = deviceMemory(), oneBuffer = largestBuffer();
  // The chain keeps four device arrays. Stay well inside what the device reports.
  auto fits = [&](size_t n) {
    const double bytes = 4.0 * (double)n;
    return n <= maxElements && !(mem > 0 && 4 * bytes > mem / 4) && !(oneBuffer > 0 && bytes > oneBuffer);
  };

  // Grow the tensor until one kernel takes long enough to time, or the size limit is reached.
  int rows = 16;
  for (;;) {
    const size_t n = (size_t)rows * d;
    Buf *X = pmpp::alloc(n * sizeof(float)), *R = pmpp::alloc(n * sizeof(float));
    const std::string failed = pmpp::take_error();
    double s = 0;
    if (failed.empty()) s = pmpp::time_it(a.reps, [&] { return runAddResidual(X, R, n); }).min_s;
    pmpp::release(X); pmpp::release(R);
    if (!failed.empty()) {
      if (rows == 16) { std::printf("bench stopped: %s\n", failed.c_str()); return 1; }
      rows /= 2;      // the last size that could be allocated
      break;
    }
    if (s > 0.05 || !fits(2 * n)) break;
    rows *= 2;
  }
  const size_t n = (size_t)rows * d;
  const double bytes = 4.0 * (double)n;
  Data D({rows, d}, 77u);

  // --- each kernel alone, on data that is already on the device ---
  std::printf("tensor: %d rows x %d columns = %zu elements (%.0f MB)\n", rows, d, n, bytes / 1e6);
  std::printf("each kernel alone, data already on the device, best of %d runs after one warm-up run\n", a.reps);
  {
    Buf *X = pmpp::alloc(n * sizeof(float)), *Y = pmpp::alloc(n * sizeof(float)),
        *B = pmpp::alloc((size_t)d * sizeof(float)), *R = pmpp::alloc(n * sizeof(float));
    std::string failed = pmpp::take_error();
    double s[3] = {0, 0, 0};
    bool deviceClock = false;
    if (failed.empty()) {
      pmpp::upload(X, D.x.data(), n * sizeof(float));
      pmpp::upload(B, D.bias.data(), (size_t)d * sizeof(float));
      pmpp::upload(R, D.r.data(), n * sizeof(float));
      s[0] = pmpp::time_it(a.reps, [&] { return runAddBias(Y, X, B, rows, d); }).min_s;
      s[1] = pmpp::time_it(a.reps, [&] { return runScaleShift(Y, X, kAlpha, kBeta, n); }).min_s;
      s[2] = pmpp::time_it(a.reps, [&] {
        const pmpp::Timing t = runAddResidual(Y, R, n);
        deviceClock = t.gpu_s >= 0;
        return t;
      }).min_s;
      failed = pmpp::take_error();
    }
    pmpp::release(X); pmpp::release(Y); pmpp::release(B); pmpp::release(R);
    if (!failed.empty() || s[0] <= 0 || s[1] <= 0 || s[2] <= 0) {
      std::printf("bench stopped: a kernel did not run (%s)\n", failed.empty() ? "no time was reported" : failed.c_str());
      return 1;
    }
    // Bytes of global memory each thread asks for: the two adds read two floats
    // and write one, scaleShift reads one and writes one. This is traffic
    // requested, not traffic observed: addBias asks for the same d bias values
    // again and again, and whether those reads reach memory every time is up to
    // the device.
    std::printf("(time: %s)\n", deviceClock ? "execution time reported by the device" : "host clock around launch and wait");
    const char* names[3] = {"addBias", "scaleShift", "addResidual"};
    const double bytesPerElement[3] = {12, 8, 12};
    for (int k = 0; k < 3; ++k) {
      const double gbps = bytesPerElement[k] * (double)n / s[k] / 1e9;
      std::printf("  %-12s %2.0f bytes requested per element  %9.4f ms  %8.1f GB/s requested\n", names[k], bytesPerElement[k],
                  s[k] * 1e3, gbps);
      pmpp::bench_row(a, "op").s("kernel", names[k]).n("elements", (double)n).n("seconds", s[k]).n("gb_per_s", gbps).emit();
    }
  }

  // --- the chain: one round trip against three ---
  std::vector<float> y = nans(n), ref(n);
  cpuBlockTail(D.x.data(), D.bias.data(), D.r.data(), kAlpha, kBeta, ref.data(), rows, d);
  auto resident = [&] { return callBlockTail(D.x.data(), D.bias.data(), D.r.data(), kAlpha, kBeta, y.data(), rows, d); };
  auto trips = [&] { return threeRoundTrips(D.x.data(), D.bias.data(), D.r.data(), kAlpha, kBeta, y.data(), rows, d); };

  // Each variant: one warm-up call whose result is checked, then the timed calls.
  int rc = resident();
  if (rc != 0) { std::printf("bench stopped: blockTail returned %d (an allocation of %.0f MB may have failed)\n", rc, bytes / 1e6); return 1; }
  std::string wrong = wrongHostResult(y, ref, d);
  if (!wrong.empty()) { std::printf("bench stopped: blockTail gave a wrong result (%s)\n", wrong.c_str()); return 1; }
  const double resident_s = bestWall(a.reps, [&] { rc |= resident(); });
  if (rc != 0) { std::printf("bench stopped: blockTail failed during the timed calls\n"); return 1; }

  y = nans(n);
  rc = trips();
  if (rc != 0) { std::printf("bench stopped: the round-trip variant could not run (an allocation of %.0f MB may have failed)\n", bytes / 1e6); return 1; }
  wrong = wrongHostResult(y, ref, d);
  if (!wrong.empty()) { std::printf("bench stopped: the round-trip variant gave a wrong result (%s)\n", wrong.c_str()); return 1; }
  const double roundtrips_s = bestWall(a.reps, [&] { rc |= trips(); });
  if (rc != 0) { std::printf("bench stopped: the round-trip variant failed during the timed calls\n"); return 1; }

  const double ratio = roundtrips_s / resident_s;
  std::printf("the chain y = alpha * (x + bias) + beta + r, host arrays in, a host array out,\n");
  std::printf("wall clock, best of %d calls after one checked warm-up call\n", a.reps);
  std::printf("  one blockTail call, your code                    %10.3f ms\n", resident_s * 1e3);
  std::printf("  three round trips, composed by the driver        %10.3f ms\n", roundtrips_s * 1e3);
  std::printf("  (each round trip: %s)\n", roundTripKind());
  std::printf("  round trips / blockTail                          %10.2fx\n", ratio);
#if defined(PMPP_BACKEND_METAL)
  std::printf("On Metal a round trip crosses no bus: it is the cost of an interface that takes host arrays,\n");
  std::printf("which have to be copied into shared buffers each time. Data kept in shared buffers is not copied at all.\n");
#endif
  std::printf("Compare the last line with what you wrote down.\n");
  pmpp::bench_row(a, "chain").n("rows", rows).n("d", d).n("n", (double)n).n("resident_s", resident_s)
      .n("roundtrips_s", roundtrips_s).n("ratio", ratio).s("roundtrip", roundTripKind()).emit();
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
