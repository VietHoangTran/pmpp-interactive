// Lab 19.1 driver: tests your unroll kernel on its own, your direct convolutional layer,
// your unroll + GEMM layer and the provided implicit GEMM against a CPU reference, then
// times the three layer paths on five layer shapes.
//   X: N x C x H x W, F: M x C x K x K, Y: N x M x H_out x W_out (NCHW), row-major float;
//   stride 1, no padding: H_out = H - K + 1, W_out = W - K + 1, P = H_out * W_out.
// You do not need to edit this file; your work goes in work/.
//
// What a test case does:
//   - X and F are allocated with exactly the floats they hold, and on the emulators the
//     scratch Bt with exactly P x C*K*K, so on `emu` a read or write past any of them is
//     an AddressSanitizer report that names the line. On a GPU, Bt is followed by a guard
//     band of sentinel floats instead.
//   - Y is followed, in the same allocation, by a guard band of sentinel floats (256 on
//     the emulators; on a GPU one whole sample's worth more), which must survive.
//   - Y and Bt start as NaN before every launch, so an element nobody wrote cannot pass.
//   - Four checks per shape: (a) the unroll alone, every element of every sample's B^T
//     compared exactly with the CPU unroll, so an index error shows up as an unroll error
//     and not as a GEMM error; (b) the direct path; (c) the unroll + GEMM path; (d) the
//     provided implicit GEMM (a failure there is the course's bug).
//   - (b) to (d) are compared with cpuConvLayer (Fig. 19.4, accumulated in double), with
//     a tolerance scaled by the sum of |x f| per output (labs 5.2 and 6.4 use the same
//     rule). The book's example must match exactly: small integers are exact in float.
//   - On a GPU every check runs twice, with Y and Bt back to NaN in between.
#include "pmpp/harness.hpp"

#include <limits>
#include <memory>

#ifndef TILE_WIDTH
#define TILE_WIDTH 16
#endif
#ifndef PMPP_CH19_GEMM_SOURCE
#define PMPP_CH19_GEMM_SOURCE "m2 reference"
#endif

using pmpp::Buf;

// ---------------------------------------------------------------- the CPU references
// [lesson:cpu-reference]
// Fig. 19.4 (p. 459), the batched layer, with its indices linearised row-major by hand and
// the sum kept in double. mag (may be null) receives, per output, the sum of |x * f| over
// its terms: the tests scale their tolerance with it.
void cpuConvLayer(int N, int M, int C, int H, int W, int K,
                  const float* X, const float* F, float* Y, float* mag) {
  const int H_out = H - K + 1, W_out = W - K + 1;
  for (int n = 0; n < N; n++)
    for (int m = 0; m < M; m++)
      for (int h = 0; h < H_out; h++)
        for (int w = 0; w < W_out; w++) {
          double acc = 0, am = 0;
          for (int c = 0; c < C; c++)
            for (int p = 0; p < K; p++)
              for (int q = 0; q < K; q++) {
                const double t = (double)X[(((size_t)n * C + c) * H + (h + p)) * W + (w + q)]   // X[n, c, h+p, w+q]
                               * F[(((size_t)m * C + c) * K + p) * K + q];                    // F[m, c, p, q]
                acc += t;
                am += std::fabs(t);
              }
          const size_t i = (((size_t)n * M + m) * H_out + h) * W_out + w;                      // Y[n, m, h, w]
          Y[i] = (float)acc;
          if (mag) mag[i] = (float)am;
        }
}
// [/lesson:cpu-reference]

// Sample n's unrolled input, transposed: Bt[v][u] = B[u][v]. Row v is output pixel
// (h, w) = (v / W_out, v % W_out); column u runs over the patches in the book's
// (c, p, q) order (pp. 465-466). Written as loops, independently of Eq. 19.5's divides.
static void cpuUnroll(const float* X, int n, int C, int H, int W, int K, std::vector<float>& Bt) {
  const int H_out = H - K + 1, W_out = W - K + 1, CKK = C * K * K;
  Bt.assign((size_t)H_out * W_out * CKK, 0.f);
  size_t i = 0;
  for (int h = 0; h < H_out; h++)
    for (int w = 0; w < W_out; w++)
      for (int c = 0; c < C; c++)
        for (int p = 0; p < K; p++)
          for (int q = 0; q < K; q++)
            Bt[i++] = X[(((size_t)n * C + c) * H + (h + p)) * W + (w + q)];
}

// ---------------------------------------------------------------- the problem
struct Shape { int N, C, H, W, M, K; const char* why; bool book; };

// Fig. 19.2 (p. 456): three 3 x 3 input maps, six 2 x 2 filters, two 2 x 2 output maps.
static const float kBookX[27] = {1, 2, 0, 1, 1, 3, 0, 2, 2,  0, 2, 1, 0, 3, 2, 1, 1, 0,  1, 2, 1, 0, 1, 3, 3, 3, 2};
static const float kBookF[24] = {1, 1, 2, 2,  1, 1, 1, 1,  0, 1, 1, 0,     // output map 0
                                 1, 0, 0, 1,  2, 1, 2, 1,  1, 2, 2, 0};    // output map 1
static const float kBookY[8] = {14, 20, 15, 24,  12, 24, 17, 26};

static const float kSentinel = -77777.0f;   // no correct kernel writes anything past Y or Bt
static const float kNaN = std::numeric_limits<float>::quiet_NaN();

static std::string scratchProblem(int C, int H, int W, int K);   // per backend, below

// A problem instance on the device. `forBench` skips the guard bands.
struct Case {
  Shape s;
  int Ho, Wo, P, CKK;
  size_t nX, nF, nY, nB, guardY, guardB;
  std::vector<float> x, f, ref, mag;
  std::vector<std::vector<float>> bref;   // CPU B^T per sample, made when first needed
  std::string scratchWhy;                 // why there is no scratch (Metal's maxBufferLength)
  std::string courseBug;                  // the CPU reference disagrees with Fig. 19.2
  Buf *X = nullptr, *F = nullptr, *Y = nullptr, *Bt = nullptr;

  explicit Case(const Shape& sh, bool forBench = false) : s(sh) {
    Ho = s.H - s.K + 1; Wo = s.W - s.K + 1; P = Ho * Wo; CKK = s.C * s.K * s.K;
    nX = (size_t)s.N * s.C * s.H * s.W;
    nF = (size_t)s.M * CKK;
    nY = (size_t)s.N * s.M * P;
    nB = (size_t)P * CKK;
    const bool emu = pmpp::is_emu();
    guardY = forBench ? 0 : emu ? 256 : (size_t)s.M * P + 1024;
    guardB = forBench || emu ? 0 : (size_t)CKK * TILE_WIDTH + 1024;
    if (s.book) {
      for (int n = 0; n < s.N; ++n) x.insert(x.end(), kBookX, kBookX + 27);
      f.assign(kBookF, kBookF + 24);
    } else {
      const uint32_t seed = 1900u + (uint32_t)(s.N * 7919 + s.C * 977 + s.H * 131 + s.W * 31 + s.M * 7 + s.K);
      x = pmpp::random_floats(nX, seed);
      f = pmpp::random_floats(nF, seed + 1);
    }
    X = pmpp::alloc(nX * 4);
    F = pmpp::alloc(nF * 4);
    Y = pmpp::alloc((nY + guardY) * 4);
    pmpp::upload(X, x.data(), nX * 4);
    pmpp::upload(F, f.data(), nF * 4);
    scratchWhy = scratchProblem(s.C, s.H, s.W, s.K);
    if (scratchWhy.empty()) Bt = pmpp::alloc((nB + guardB) * 4);   // else: not allocated
  }
  Case(const Case&) = delete;
  ~Case() { pmpp::release(X); pmpp::release(F); pmpp::release(Y); if (Bt) pmpp::release(Bt); }

  double flop() const { return 2.0 * s.N * (double)s.M * CKK * P; }

  void refillY() {
    std::vector<float> v(nY + guardY, kNaN);
    std::fill(v.begin() + nY, v.end(), kSentinel);
    pmpp::upload(Y, v.data(), v.size() * 4);
  }
  void refillBt() {
    std::vector<float> v(nB + guardB, kNaN);
    std::fill(v.begin() + nB, v.end(), kSentinel);
    pmpp::upload(Bt, v.data(), v.size() * 4);
  }

  void reference() {
    if (!ref.empty()) return;
    ref.resize(nY);
    mag.resize(nY);
    cpuConvLayer(s.N, s.M, s.C, s.H, s.W, s.K, x.data(), f.data(), ref.data(), mag.data());
    if (s.book)
      for (size_t i = 0; i < nY; ++i)
        if (ref[i] != kBookY[i % 8]) courseBug = "the course's CPU reference does not reproduce Fig. 19.2: please report it";
  }

  // Empty if Y matches the reference (exactly, for the book's example) and the guard
  // band is untouched.
  std::string checkY() {
    reference();
    if (!courseBug.empty()) return courseBug;
    std::vector<float> got(nY + guardY);
    pmpp::download(Y, got.data(), got.size() * 4);
    const double rtol = s.book ? 0.0 : 1e-4;
    size_t bad = 0, nan = 0, first = 0;
    double maxErr = 0;
    for (size_t i = 0; i < nY; ++i) {
      const double e = std::fabs((double)got[i] - (double)ref[i]);
      if (!(e <= rtol * (1.0 + mag[i]))) {
        if (bad == 0) first = i;
        ++bad;
        if (std::isnan(got[i])) ++nan;
      }
      if (e > maxErr) maxErr = e;
    }
    std::string err;
    char b[600];
    if (bad) {
      const size_t w = first % Wo, h = (first / Wo) % Ho, m = (first / P) % s.M, n = first / ((size_t)P * s.M);
      std::snprintf(b, sizeof b, "%zu of %zu outputs wrong (%zu still NaN: never written); first Y[n %zu, m %zu, h %zu, w %zu]: "
                    "got %.7g, expected %.7g", bad, nY, nan, n, m, h, w, (double)got[first], (double)ref[first]);
      err = b;
      if (nan < bad) { std::snprintf(b, sizeof b, "; largest error %.3g", maxErr); err += b; }
      if (s.book) {
        err += "; sample 0 came out";
        for (int i = 0; i < 8; ++i) { std::snprintf(b, sizeof b, "%s %g", i == 4 ? " /" : "", (double)got[i]); err += b; }
        err += ", Fig. 19.2 has 14 20 15 24 / 12 24 17 26";
      }
    }
    size_t changed = 0, firstG = 0;
    for (size_t i = nY; i < got.size(); ++i)
      if (std::memcmp(&got[i], &kSentinel, 4) != 0) { if (!changed) firstG = i - nY; ++changed; }
    if (changed) {
      std::snprintf(b, sizeof b, "%s%zu of the %zu floats after Y were overwritten, the first %zu floats past its end",
                    err.empty() ? "" : "; ", changed, guardY, firstG);
      err += b;
    }
    return err;
  }

  // Empty if Bt holds exactly sample n's B^T and its guard band (GPU only) is untouched.
  std::string checkUnroll(int n) {
    if (bref.empty()) bref.resize(s.N);
    if (bref[n].empty()) cpuUnroll(x.data(), n, s.C, s.H, s.W, s.K, bref[n]);
    const std::vector<float>& want = bref[n];
    std::vector<float> got(nB + guardB);
    pmpp::download(Bt, got.data(), got.size() * 4);
    size_t bad = 0, nan = 0, first = 0;
    for (size_t i = 0; i < nB; ++i)
      if (!(got[i] == want[i])) { if (!bad) first = i; ++bad; if (std::isnan(got[i])) ++nan; }
    std::string err;
    char b[600];
    if (bad) {
      const int v = (int)(first / CKK), u = (int)(first % CKK);
      const int h = v / Wo, w = v % Wo, c = u / (s.K * s.K), p = (u % (s.K * s.K)) / s.K, q = u % s.K;
      std::snprintf(b, sizeof b, "sample %d: %zu of %zu elements of B^T wrong (%zu still NaN: never written); first at "
                    "row %d (output pixel h %d, w %d), column %d (c %d, p %d, q %d): got %.7g, expected %.7g = "
                    "X[n %d, c %d, h %d, w %d]", n, bad, nB, nan, v, h, w, u, c, p, q, (double)got[first],
                    (double)want[first], n, c, h + p, w + q);
      err = b;
    }
    size_t changed = 0;
    for (size_t i = nB; i < got.size(); ++i)
      if (std::memcmp(&got[i], &kSentinel, 4) != 0) ++changed;
    if (changed) {
      std::snprintf(b, sizeof b, "%ssample %d: %zu of the %zu floats after Bt were overwritten", err.empty() ? "" : "; ",
                    n, changed, guardB);
      err += b;
    }
    return err;
  }
};

// ---------------------------------------------------------------- backends
// Each backend section defines, for the three paths and the unroll:
//   directProblem() ... implicitProblem()   empty if that code can run here, else why not
//   runDirect, runUnroll (one sample), runConvGemm, runImplicit
//   runUnrollAll, runGemmAll                the benchmark's two halves of unroll + GEMM
//   scratchProblem(C, H, W, K)              Metal: launch.mm's check against maxBufferLength
//   memoryBudget(), oneBufferLimit()        what the device reports, for sizing the bench
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeConvDirect(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> F,
                      id<MTLBuffer> Y, int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K,
                      int32_t tileWidth);
void encodeUnroll(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, int32_t n,
                  id<MTLBuffer> Bt, int32_t C, int32_t H, int32_t W, int32_t K);
void encodeConvGemm(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> unrollPipeline,
                    id<MTLComputePipelineState> linearPipeline, id<MTLBuffer> X, id<MTLBuffer> F, id<MTLBuffer> Y,
                    id<MTLBuffer> Bt, int32_t N, int32_t C, int32_t H, int32_t W, int32_t M, int32_t K,
                    int32_t tileWidth);
std::string convGemmScratchProblem(id<MTLDevice>, int32_t C, int32_t H, int32_t W, int32_t K);   // launch.mm
void encodeLinearAt(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, NSUInteger xOff,
                    id<MTLBuffer> W, NSUInteger wOff, id<MTLBuffer> Y, NSUInteger yOff, int32_t batch, int32_t in,
                    int32_t out, int32_t tileWidth);                                   // provided/linear_at.mm
void encodeConvImplicitGemm(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X,
                            id<MTLBuffer> F, id<MTLBuffer> Y, int32_t N, int32_t C, int32_t H, int32_t W,
                            int32_t M, int32_t K, int32_t tileWidth);   // provided/implicit_gemm_launch.mm
using pmpp::metal::buffer;

// <lab>/<variant>/kernel.metal -> <lab>
static std::string labDir() {
  std::string s = PMPP_METAL_SOURCE;
  s = s.substr(0, s.rfind('/'));
  return s.substr(0, s.rfind('/'));
}
static const std::vector<std::pair<std::string, std::string>>& tileMacro() {
  static const std::vector<std::pair<std::string, std::string>> m = {{"TILE_WIDTH", std::to_string(TILE_WIDTH)}};
  return m;
}
static id<MTLComputePipelineState> directPipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "conv_direct", tileMacro()); }
static id<MTLComputePipelineState> unrollPipeline() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "unroll_input", tileMacro()); }
static id<MTLComputePipelineState> implicitPipeline() {
  return pmpp::metal::pipeline(labDir() + "/provided/implicit_gemm.metal", "conv_implicit_gemm", tileMacro());
}
#if defined(PMPP_CH19_M2_METAL)
static id<MTLComputePipelineState> linearPipeline() { return pmpp::metal::pipeline(PMPP_CH19_M2_METAL, "linear", tileMacro()); }
static const char* linearWhere() { return PMPP_CH19_M2_METAL; }
#else
static id<MTLComputePipelineState> linearPipeline() { return nil; }
static const char* linearWhere() { return "(not configured: provided/gemm.cmake did not run)"; }
#endif

// Empty if the pipeline compiled and can run `threads` threads per threadgroup.
static std::string pipelineProblem(id<MTLComputePipelineState> p, NSUInteger threads, const std::string& what) {
  if (!p) return what + " did not compile: " + pmpp::take_error();
  if (p.maxTotalThreadsPerThreadgroup < threads)
    return what + ": this pipeline allows " + std::to_string((unsigned long)p.maxTotalThreadsPerThreadgroup) +
           " threads per threadgroup, the kernel needs " + std::to_string((unsigned long)threads);
  return "";
}
static const NSUInteger kTile2 = (NSUInteger)TILE_WIDTH * TILE_WIDTH;
static std::string unrollProblem() { return pipelineProblem(unrollPipeline(), 1, "your unroll_input"); }
static std::string directProblem() { return pipelineProblem(directPipeline(), kTile2, "your conv_direct"); }
static std::string gemmProblem() {
  std::string e = unrollProblem();
  if (e.empty()) e = pipelineProblem(linearPipeline(), kTile2, std::string("lab 6.4's linear kernel, ") + linearWhere());
  return e;
}
static std::string implicitProblem() {
  return pipelineProblem(implicitPipeline(), kTile2, "the provided conv_implicit_gemm");
}
static std::string scratchProblem(int C, int H, int W, int K) {
  return convGemmScratchProblem(pmpp::metal::device(), C, H, W, K);
}

static pmpp::Timing runDirect(Case& c) {
  id<MTLComputePipelineState> p = directPipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeConvDirect(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), buffer(c.F), buffer(c.Y), c.s.N, c.s.C, c.s.H,
                     c.s.W, c.s.M, c.s.K, TILE_WIDTH);
  });
}
static pmpp::Timing runUnroll(Case& c, int n) {
  id<MTLComputePipelineState> p = unrollPipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeUnroll(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), n, buffer(c.Bt), c.s.C, c.s.H, c.s.W, c.s.K);
  });
}
static pmpp::Timing runConvGemm(Case& c) {
  id<MTLComputePipelineState> u = unrollPipeline(), l = linearPipeline();
  if (!u || !l) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeConvGemm(((pmpp::metal::Pass*)v)->encoder, u, l, buffer(c.X), buffer(c.F), buffer(c.Y), buffer(c.Bt), c.s.N,
                   c.s.C, c.s.H, c.s.W, c.s.M, c.s.K, TILE_WIDTH);
  });
}
static pmpp::Timing runImplicit(Case& c) {
  id<MTLComputePipelineState> p = implicitPipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    encodeConvImplicitGemm(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), buffer(c.F), buffer(c.Y), c.s.N, c.s.C,
                           c.s.H, c.s.W, c.s.M, c.s.K, TILE_WIDTH);
  });
}
static pmpp::Timing runUnrollAll(Case& c) {
  id<MTLComputePipelineState> p = unrollPipeline();
  if (!p) return {0, 0};
  return pmpp::run([&](void* v) {
    for (int n = 0; n < c.s.N; ++n)
      encodeUnroll(((pmpp::metal::Pass*)v)->encoder, p, buffer(c.X), n, buffer(c.Bt), c.s.C, c.s.H, c.s.W, c.s.K);
  });
}
static pmpp::Timing runGemmAll(Case& c) {
  id<MTLComputePipelineState> l = linearPipeline();
  if (!l) return {0, 0};
  return pmpp::run([&](void* v) {
    for (int n = 0; n < c.s.N; ++n)
      encodeLinearAt(((pmpp::metal::Pass*)v)->encoder, l, buffer(c.F), 0, buffer(c.Bt), 0, buffer(c.Y),
                     (NSUInteger)n * c.s.M * c.P * sizeof(float), c.s.M, c.CKK, c.P, TILE_WIDTH);
  });
}
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* memoryBudgetName() { return "recommendedMaxWorkingSetSize"; }
static double oneBufferLimit() { return pmpp::device_prop("maxBufferLength", 0); }
#else
void launchConvDirect(const float* X, const float* F, float* Y, int N, int C, int H, int W, int M, int K);
void launchUnroll(const float* X, int n, float* Bt, int C, int H, int W, int K);
void launchConvGemm(const float* X, const float* F, float* Y, float* Bt, int N, int C, int H, int W, int M, int K);
void launchLinear(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 6.4's
void launchConvImplicitGemm(const float* X, const float* F, float* Y, int N, int C, int H, int W, int M,
                            int K);                                                       // provided/implicit_gemm.cu
static const float* dIn(Buf* b) { return (const float*)pmpp::raw(b); }
static float* dOut(Buf* b) { return (float*)pmpp::raw(b); }

static std::string blockProblem(const char* what) {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  if (TILE_WIDTH * TILE_WIDTH > limit)
    return std::string(what) + ": this device allows " + std::to_string(limit) + " threads per block";
  return "";
}
static std::string unrollProblem() { return ""; }
static std::string directProblem() { return blockProblem("the direct kernel's tile"); }
static std::string gemmProblem() { return blockProblem("lab 6.4's tile"); }
static std::string implicitProblem() { return blockProblem("the implicit GEMM's tile"); }
static std::string scratchProblem(int, int, int, int) { return ""; }

static pmpp::Timing runDirect(Case& c) {
  return pmpp::run([&](void*) { launchConvDirect(dIn(c.X), dIn(c.F), dOut(c.Y), c.s.N, c.s.C, c.s.H, c.s.W, c.s.M, c.s.K); });
}
static pmpp::Timing runUnroll(Case& c, int n) {
  return pmpp::run([&](void*) { launchUnroll(dIn(c.X), n, dOut(c.Bt), c.s.C, c.s.H, c.s.W, c.s.K); });
}
static pmpp::Timing runConvGemm(Case& c) {
  return pmpp::run([&](void*) {
    launchConvGemm(dIn(c.X), dIn(c.F), dOut(c.Y), dOut(c.Bt), c.s.N, c.s.C, c.s.H, c.s.W, c.s.M, c.s.K);
  });
}
static pmpp::Timing runImplicit(Case& c) {
  return pmpp::run([&](void*) {
    launchConvImplicitGemm(dIn(c.X), dIn(c.F), dOut(c.Y), c.s.N, c.s.C, c.s.H, c.s.W, c.s.M, c.s.K);
  });
}
static pmpp::Timing runUnrollAll(Case& c) {
  return pmpp::run([&](void*) {
    for (int n = 0; n < c.s.N; ++n) launchUnroll(dIn(c.X), n, dOut(c.Bt), c.s.C, c.s.H, c.s.W, c.s.K);
  });
}
static pmpp::Timing runGemmAll(Case& c) {
  return pmpp::run([&](void*) {
    for (int n = 0; n < c.s.N; ++n)
      launchLinear(dIn(c.F), dIn(c.Bt), dOut(c.Y) + (size_t)n * c.s.M * c.P, c.s.M, c.CKK, c.P);
  });
}
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* memoryBudgetName() { return "totalGlobalMem"; }
static double oneBufferLimit() { return 0; }   // no per-allocation cap below the memory size is checked here
#endif

static bool gemmIsYours() { return std::string(PMPP_CH19_GEMM_SOURCE) == "m2 yours"; }

// ---------------------------------------------------------------- tests
static std::vector<Shape> testShapes() {
  const Shape book1 = {1, 3, 3, 3, 2, 2, "the book's example, Fig. 19.2: Y must be exactly 14 20 15 24 / 12 24 17 26", true};
  const Shape nonsq = {1, 2, 9, 17, 3, 3, "not square: a 7 x 15 output", false};
  if (std::string(pmpp::backend()) == "race")   // ThreadSanitizer is slow: three small shapes
    return {book1, nonsq, {1, 6, 14, 14, 16, 5, "LeNet C3, one sample: C*K*K = 150, a 10 x 10 output", false}};
  std::vector<Shape> s = {
      book1,
      {2, 3, 3, 3, 2, 2, "the book's example twice: a batch of two", true},
      {2, 1, 32, 32, 6, 5, "LeNet C1: C*K*K = 25, a 28 x 28 output, M = 6; none a multiple of 16", false},
      {2, 6, 14, 14, 16, 5, "LeNet C3: C*K*K = 150, a 10 x 10 output", false},
      {2, 8, 7, 9, 5, 1, "pointwise, K = 1: B is X itself, C x H*W", false},
      {1, 4, 5, 5, 3, 5, "one output pixel: P = 1", false},
      nonsq,
      {1, 16, 6, 6, 16, 2, "C*K*K = 64, a multiple of 16", false},
  };
  if (!pmpp::is_emu()) {
    s.push_back({4, 64, 56, 56, 64, 3, "a middle layer at batch 4", false});
    s.push_back({8, 256, 14, 14, 256, 3, "a late layer at batch 8", false});
    s.push_back({2, 3, 224, 224, 64, 3, "an early layer at batch 2", false});
  }
  return s;
}

static std::string shapeName(const Shape& s) {
  char b[300];
  std::snprintf(b, sizeof b, "N %d, C %d, H %d, W %d, M %d, K %d  (%s)", s.N, s.C, s.H, s.W, s.M, s.K, s.why);
  return b;
}

static int repeats() { return pmpp::is_emu() ? 1 : 2; }

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("GEMM for unroll + GEMM: %s (lab 6.4's launchLinear; PMPP_CH19_GEMM=work selects your own)\n",
              PMPP_CH19_GEMM_SOURCE);
  // A path whose code cannot run here (a compile error, a block too large) fails once,
  // not once per shape.
  const std::string probUnroll = unrollProblem(), probDirect = directProblem(), probGemm = gemmProblem(),
                    probImplicit = implicitProblem();
  if (!probUnroll.empty()) t.check("your unroll kernel runs here", [&] { return probUnroll; });
  if (!probDirect.empty()) t.check("your direct kernel runs here", [&] { return probDirect; });
  if (!probGemm.empty() && probUnroll.empty()) t.check("the GEMM path runs here", [&] { return probGemm; });
  if (!probImplicit.empty())
    t.check("the provided implicit GEMM runs here", [&] { return "the course's bug, not yours: " + probImplicit; });

  for (const Shape& s : testShapes()) {
    Case c(s);
    const std::string name = shapeName(s);
    bool unrollPassed = false;

    if (probUnroll.empty())
      t.check("(a) unroll alone: " + name, [&]() -> std::string {
        unrollPassed = false;
        if (!c.Bt) return c.scratchWhy;
        for (int r = 0; r < repeats(); ++r)
          for (int n = 0; n < s.N; ++n) {
            c.refillBt();
            runUnroll(c, n);
            const std::string e = c.checkUnroll(n);
            if (!e.empty()) return e;
          }
        unrollPassed = true;
        return "";
      });

    if (probDirect.empty())
      t.check("(b) direct: " + name, [&]() -> std::string {
        for (int r = 0; r < repeats(); ++r) {
          c.refillY();
          runDirect(c);
          const std::string e = c.checkY();
          if (!e.empty()) return e;
        }
        return "";
      });

    if (probGemm.empty())
      t.check("(c) unroll + GEMM: " + name, [&]() -> std::string {
        if (!c.Bt) return c.scratchWhy;
        for (int r = 0; r < repeats(); ++r) {
          c.refillY();
          c.refillBt();
          runConvGemm(c);
          std::string e = c.checkY();
          if (!e.empty()) {
            if (gemmIsYours())
              e += ". The GEMM is your lab 6.4 kernel (PMPP_CH19_GEMM=work): check that lab 6.4 passes first";
            if (unrollPassed)
              e += std::string(gemmIsYours() ? "; then, since" : ". Since") +
                   " the unroll alone passed on this shape, look at what launchConvGemm passes to the GEMM "
                   "(which matrix is X, which is W, the three sizes, and where sample n's Y starts)";
            return e;
          }
        }
        return "";
      });

    if (probImplicit.empty())
      t.check("(d) provided implicit GEMM: " + name, [&]() -> std::string {
        for (int r = 0; r < repeats(); ++r) {
          c.refillY();
          runImplicit(c);
          const std::string e = c.checkY();
          if (!e.empty()) return "the provided implicit GEMM is wrong here; this is the course's bug, not yours: " + e;
        }
        return "";
      });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
struct Layer { const char* name; int N, C, H, W, M, K; };

static double expansion(const Shape& s) {   // Eq. 19.4: K^2 H_out W_out / (H W), computed
  const double Ho = s.H - s.K + 1, Wo = s.W - s.K + 1;
  return (double)s.K * s.K * Ho * Wo / ((double)s.H * s.W);
}

static int bench(const pmpp::Args& a) {
  const std::vector<Layer> layers = {
      {"early", 8, 3, 224, 224, 64, 3},
      {"middle", 8, 64, 56, 56, 64, 3},
      {"late", 8, 256, 14, 14, 256, 3},
      {"pointwise", 8, 256, 14, 14, 256, 1},
      {"small", 64, 6, 14, 14, 16, 5},   // LeNet C3 at batch 64
  };
  const int reps = std::max(a.reps, 1);
  std::printf("\nbenchmark: three paths per layer, taking turns; best of %d rounds after a warm-up\n", reps);
  std::printf("GEMM: %s (lab 6.4's launchLinear, TILE_WIDTH %d)\n", PMPP_CH19_GEMM_SOURCE, TILE_WIDTH);
  const double budget = memoryBudget(), oneBuf = oneBufferLimit();
  if (budget > 0) std::printf("%s of this device: %.0f bytes; a layer may use at most half\n", memoryBudgetName(), budget);
  if (oneBuf > 0) std::printf("maxBufferLength of this device: %.0f bytes (the most one buffer, and so the scratch, may hold)\n", oneBuf);

  // A timing of a wrong kernel means nothing: each path is checked on two awkward shapes first.
  const Shape gate[2] = {{2, 1, 32, 32, 6, 5, "LeNet C1", false}, {1, 2, 9, 17, 3, 3, "not square", false}};
  const char* names[3] = {"direct", "unroll_gemm", "implicit"};
  std::string why[3] = {directProblem(), gemmProblem(), implicitProblem()};
  for (const Shape& g : gate) {
    Case c(g);
    for (int k = 0; k < 3; ++k) {
      if (!why[k].empty()) continue;
      if (k == 1 && !c.Bt) { why[k] = c.scratchWhy; continue; }
      c.refillY();
      if (k == 1) c.refillBt();
      if (k == 0) runDirect(c); else if (k == 1) runConvGemm(c); else runImplicit(c);
      const std::string e = c.checkY() + pmpp::take_error();
      if (!e.empty()) why[k] = std::string("wrong on ") + g.why + ": " + e;
    }
  }
  int rc = 0;
  for (int k = 0; k < 3; ++k)
    if (!why[k].empty()) {
      std::printf("%s: not timed (%s)\n", names[k], why[k].c_str());
      if (k != 2) rc = 1;
    }

  for (const Layer& L : layers) {
    const Shape s = {L.N, L.C, L.H, L.W, L.M, L.K, L.name, false};
    const int Ho = s.H - s.K + 1, Wo = s.W - s.K + 1, P = Ho * Wo, CKK = s.C * s.K * s.K;
    const double nX = (double)s.N * s.C * s.H * s.W, nY = (double)s.N * s.M * P, nB = (double)P * CKK;
    const double bytes = 4.0 * (nX + (double)s.M * CKK + nY + nB);
    std::printf("\n%s: N %d, C %d, %d x %d -> %d x %d, M %d, K %d\n", L.name, s.N, s.C, s.H, s.W, Ho, Wo, s.M, s.K);
    std::printf("  per-sample GEMM M x C*K*K x P = %d x %d x %d; expansion (Eq. 19.4, computed) %.2f; "
                "scratch %.0f bytes per sample\n", s.M, CKK, P, expansion(s), 4.0 * nB);
    if (oneBuf > 0) std::printf("  scratch %.0f bytes beside maxBufferLength %.0f bytes\n", 4.0 * nB, oneBuf);
    if (budget > 0 && bytes > budget / 2) {
      std::printf("  skipped: needs %.0f bytes, more than half of %s\n", bytes, memoryBudgetName());
      continue;
    }
    if (oneBuf > 0 && 4.0 * std::max(nX, std::max(nY, nB)) > oneBuf) {
      std::printf("  skipped: one of its buffers is larger than maxBufferLength\n");
      continue;
    }
    Case c(s, true);
    const bool run[3] = {why[0].empty(), why[1].empty() && c.Bt != nullptr, why[2].empty()};
    if (why[1].empty() && !c.Bt) std::printf("  unroll + GEMM: %s\n", c.scratchWhy.c_str());

    // Five timed things take turns: direct, the whole unroll + GEMM layer, its N unrolls
    // alone, its N GEMMs alone (on the scratch the previous run left filled), implicit.
    std::vector<std::function<pmpp::Timing()>> fn = {
        [&] { return runDirect(c); }, [&] { return runConvGemm(c); }, [&] { return runUnrollAll(c); },
        [&] { return runGemmAll(c); }, [&] { return runImplicit(c); }};
    const bool on[5] = {run[0], run[1], run[1], run[1], run[2]};
    double best[5] = {0, 0, 0, 0, 0};
    for (int i = 0; i < 5; ++i) if (on[i]) fn[i]();   // warm-up
    for (int r = 0; r < reps; ++r)
      for (int i = 0; i < 5; ++i) {
        if (!on[i]) continue;
        const double t = fn[i]().best();
        if (r == 0 || t < best[i]) best[i] = t;
      }
    const std::string err = pmpp::take_error();
    if (!err.empty()) { std::printf("  failed: %s\n", err.c_str()); rc = 1; continue; }

    const double flop = c.flop();
    auto emit = [&](const char* name, double sec) {
      pmpp::Json j = pmpp::bench_row(a, name);
      j.s("shape", L.name).n("n", s.N).n("c", s.C).n("h", s.H).n("w", s.W).n("m", s.M).n("k", s.K)
       .n("seconds", sec).n("gflops", flop / sec / 1e9);
      if (std::string(name) == "unroll_gemm")
        j.n("unroll_seconds", best[2]).n("gemm_seconds", best[3]).n("scratch_bytes", 4.0 * nB)
         .n("expansion", expansion(s)).s("gemm_source", PMPP_CH19_GEMM_SOURCE);
      j.emit();
    };
    std::printf("  %-15s %12s %12s\n", "path", "ms", "GFLOP/s");
    if (run[0] && best[0] > 0) {
      std::printf("  %-15s %12.4f %12.1f\n", "direct", best[0] * 1e3, flop / best[0] / 1e9);
      emit("direct", best[0]);
    }
    if (run[1] && best[1] > 0) {
      std::printf("  %-15s %12.4f %12.1f   N unrolls alone %.4f ms (%.0f%% of the layer), N GEMMs alone %.4f ms\n",
                  "unroll + GEMM", best[1] * 1e3, flop / best[1] / 1e9, best[2] * 1e3, 100.0 * best[2] / best[1],
                  best[3] * 1e3);
      emit("unroll_gemm", best[1]);
    }
    if (run[2] && best[4] > 0) {
      std::printf("  %-15s %12.4f %12.1f\n", "implicit GEMM", best[4] * 1e3, flop / best[4] / 1e9);
      emit("implicit", best[4]);
    }
    if (run[0] && run[1] && best[0] > 0 && best[1] > 0) {
      const double r = best[0] / best[1];
      std::printf("  unroll + GEMM against direct: %.2fx %s\n", r >= 1 ? r : 1 / r, r >= 1 ? "faster" : "slower");
    }
  }
  std::printf("\nThese are measurements of this machine, not expectations.\n");
  return rc;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
