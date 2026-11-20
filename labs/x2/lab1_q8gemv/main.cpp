// Lab X2.1 driver: checks the CPU quantiser on hand-derived fixtures, tests your Q8_0
// matrix-vector product against a CPU reference, then times it against lab 22.1's FP32
// gather on decode-sized layers.
//   Y[b][j] = sum over k of X[b][k] * d[j][k / 32] * q[j][k],   1 <= batch <= 8, in % 32 == 0,
//   codes Wq [out x in] int8, scales Wd [out x in/32] FP16 bits, X and Y float32, batch first.
// You do not need to edit this file; your work goes in work/.
//
// What a test case does (README, "What the tests do"):
//   - The weights of each 32-weight block are scaled by a power of two between 2^-12 and
//     2^12 that changes from block to block and row to row, so a scale read from the wrong
//     block gives a wrong answer. Every block holds both +peak and -peak, so codes 127 and
//     -127 occur in every block. One row is all zero (d = 0).
//   - X, Wq and Wd are allocated with exactly the bytes they hold, so on `emu` a read past
//     any of them is an AddressSanitizer report naming the line.
//   - Y starts as NaN and is followed, in the same allocation, by a guard band of sentinels
//     that must survive. Each case launches twice without resetting Y.
//   - Results are compared with y = x dequant(W)^T computed in double, within
//     1e-4 x (1 + sum_k |x w|) for each output. Never bitwise.
#include "pmpp/harness.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <limits>
#include <map>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "../shared/qfixture.h"
#include "../shared/qformat.h"

using pmpp::Buf;

static const int kMaxBatch = 8;   // MAX_BATCH in the kernels

// ---------------------------------------------------------------- the CPU reference
// Y[b][o] = sum_k X[b][k] * w[o][k] with w = dequantize_row_q8_0(W), accumulated in double;
// mag = sum_k |X[b][k] * w[o][k]|, which scales the tolerance.
static void cpuGemvDequant(const std::vector<float>& x, const std::vector<float>& w, int batch, int in, int out,
                           std::vector<float>& y, std::vector<float>& mag) {
  y.assign((size_t)batch * out, 0.f);
  mag.assign((size_t)batch * out, 0.f);
  for (int b = 0; b < batch; ++b) {
    const float* xr = x.data() + (size_t)b * in;
    for (int o = 0; o < out; ++o) {
      const float* wr = w.data() + (size_t)o * in;
      double acc = 0, am = 0;
      for (int k = 0; k < in; ++k) { const double p = (double)xr[k] * wr[k]; acc += p; am += std::fabs(p); }
      y[(size_t)b * out + o] = (float)acc;
      mag[(size_t)b * out + o] = (float)am;
    }
  }
}

// Q8_0 for a whole matrix, row by row: codes [out x in], scales [out x in/32].
static void quantizeQ8_0(const std::vector<float>& w, int in, int out, std::vector<int8_t>& q, std::vector<uint16_t>& d) {
  q.resize((size_t)out * in);
  d.resize((size_t)out * (in / QK8_0));
  for (int o = 0; o < out; ++o)
    quantizeRowQ8_0Ref(w.data() + (size_t)o * in, q.data() + (size_t)o * in, d.data() + (size_t)o * (in / QK8_0), in);
}

// ---------------------------------------------------------------- backends
// Each backend section defines:
//   q8Problem(), f32Problem()                 empty if that kernel can run here, else why not
//   enqueueQ8(pass, X, Wq, Wd, Y, ...)        one launch of your kernel
//   enqueueF32(pass, X, W, Y, ...)            one launch of lab 22.1's gather (the baseline)
//   warpWidth(), widthSource()                the warp (SIMD-group) width, and where it came from
//   memoryBudget()                            bytes the benchmark may plan around
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeGemvQ8_0(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> X,
                    id<MTLBuffer> Wq, id<MTLBuffer> Wd, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvGather(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                      id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
// PMPP_METAL_SOURCE is labs/x2/lab1_q8gemv/<variant>/kernel.metal.
static const std::string kLabs = dirOf(dirOf(dirOf(dirOf(PMPP_METAL_SOURCE))));
static const std::string kGatherSource = kLabs + "/ch22/lab1_gemv/solution/kernel.metal";

static id<MTLComputePipelineState> q8Pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gemv_q8_0"); }
static id<MTLComputePipelineState> f32Pso() { return pmpp::metal::pipeline(kGatherSource, "gemv_gather"); }
static std::string pipelineProblem(id<MTLComputePipelineState> p, const std::string& what) {
  if (!p) return what + " did not build: " + pmpp::take_error();
  if (p.threadExecutionWidth == 0 || p.maxTotalThreadsPerThreadgroup < p.threadExecutionWidth)
    return what + ": this pipeline cannot run one whole SIMD-group per threadgroup";
  return "";
}
static std::string q8Problem() { return pipelineProblem(q8Pso(), "gemv_q8_0 in your kernel.metal"); }
static std::string f32Problem() { return pipelineProblem(f32Pso(), "gemv_gather in " + kGatherSource); }
static void enqueueQ8(void* vp, Buf* X, Buf* Wq, Buf* Wd, Buf* Y, int batch, int in, int out) {
  id<MTLComputePipelineState> p = q8Pso();
  if (!p) return;
  encodeGemvQ8_0(((pmpp::metal::Pass*)vp)->encoder, p, buffer(X), buffer(Wq), buffer(Wd), buffer(Y),
                 (uint32_t)batch, (uint32_t)in, (uint32_t)out);
}
static void enqueueF32(void* vp, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  id<MTLComputePipelineState> p = f32Pso();
  if (!p) return;
  encodeGemvGather(((pmpp::metal::Pass*)vp)->encoder, p, buffer(X), buffer(W), buffer(Y), (uint32_t)batch,
                   (uint32_t)in, (uint32_t)out);
}
// Apple documents no fixed SIMD-group width; the pipeline reports it (ch03-facts C9).
static int warpWidth() {
  id<MTLComputePipelineState> p = q8Pso();
  if (!p) p = f32Pso();
  return p ? (int)p.threadExecutionWidth : 0;
}
static std::string widthSource() { return "threadExecutionWidth of the gemv_q8_0 pipeline"; }
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }

#else  // CUDA, HIP and the emulators
void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in, int out);
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 22.1

// Blocks of 4 warps (WARPS_PER_BLOCK in kernel.cu).
static std::string blockProblem() {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  const int width = (int)pmpp::device_prop("warpSize", 0);
  if (width <= 0) return "the device reports no warpSize";
  if (limit < 4 * width) return "this device allows " + std::to_string(limit) + " threads per block; the kernels need 4 warps";
  return "";
}
static std::string q8Problem() { return blockProblem(); }
static std::string f32Problem() { return blockProblem(); }
static void enqueueQ8(void*, Buf* X, Buf* Wq, Buf* Wd, Buf* Y, int batch, int in, int out) {
  launchGemvQ8_0((const float*)pmpp::raw(X), (const int8_t*)pmpp::raw(Wq), (const uint16_t*)pmpp::raw(Wd),
                 (float*)pmpp::raw(Y), batch, in, out);
}
static void enqueueF32(void*, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  launchGemvGather((const float*)pmpp::raw(X), (const float*)pmpp::raw(W), (float*)pmpp::raw(Y), batch, in, out);
}
static int warpWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static std::string widthSource() { return "warpSize from the device properties"; }
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

static pmpp::Timing runQ8(Buf* X, Buf* Wq, Buf* Wd, Buf* Y, int batch, int in, int out) {
  return pmpp::run([&](void* p) { enqueueQ8(p, X, Wq, Wd, Y, batch, in, out); });
}

// ---------------------------------------------------------------- test weights
// Uniform values in [-1, 1], then for block i of row j: times 2^e with
// e = ((3j + 5i) mod 25) - 12, from 2^-12 to 2^12, so neighbouring blocks differ by 2^5 and
// neighbouring rows by 2^3. Weight (i mod 32) of the block is set to +2^e and weight
// ((i + 16) mod 32) to -2^e: both extremes, so codes 127 and -127 occur in every block.
// With out >= 3, row out / 2 is all zero (amax = 0, so d = 0 and every code 0).
static std::vector<float> testWeights(int in, int out, uint32_t seed, int* zeroRow) {
  std::vector<float> w = pmpp::random_floats((size_t)out * in, seed);
  const int nb = in / QK8_0;
  *zeroRow = out >= 3 ? out / 2 : -1;
  for (int j = 0; j < out; ++j) {
    for (int i = 0; i < nb; ++i) {
      float* p = w.data() + (size_t)j * in + (size_t)i * QK8_0;
      const float s = std::ldexp(1.0f, ((3 * j + 5 * i) % 25) - 12);
      for (int k = 0; k < QK8_0; ++k) p[k] *= s;
      p[i % QK8_0] = s;
      p[(i + 16) % QK8_0] = -s;
      if (j == *zeroRow) std::fill(p, p + QK8_0, 0.0f);
    }
  }
  return w;
}

// ---------------------------------------------------------------- a test problem on the device
static const float kSentinel = -77777.0f;   // no correct kernel writes anything past Y

struct Shape { int batch, in, out; const char* why; };

struct Case {
  int batch, in, out, zeroRow = -1;
  size_t guard;
  std::vector<float> x, w;      // w: the float weights before quantisation
  std::vector<int8_t> q;
  std::vector<uint16_t> d;
  const std::vector<float>* ref = nullptr;
  const std::vector<float>* mag = nullptr;
  Buf *X = nullptr, *Wq = nullptr, *Wd = nullptr, *Y = nullptr;

  explicit Case(const Shape& s) : batch(s.batch), in(s.in), out(s.out) {
    guard = pmpp::is_emu() ? 256 : (size_t)kMaxBatch * out + 256;
    const uint32_t seed = 92000u + (uint32_t)batch * 131u + (uint32_t)in * 17u + (uint32_t)out;
    x = pmpp::random_floats((size_t)batch * in, seed);
    w = testWeights(in, out, seed + 1, &zeroRow);
    quantizeQ8_0(w, in, out, q, d);
    X = pmpp::alloc(x.size() * 4);
    Wq = pmpp::alloc(q.size());
    Wd = pmpp::alloc(d.size() * 2);
    Y = pmpp::alloc(((size_t)batch * out + guard) * 4);
    pmpp::upload(X, x.data(), x.size() * 4);
    pmpp::upload(Wq, q.data(), q.size());
    pmpp::upload(Wd, d.data(), d.size() * 2);
  }
  Case(const Case&) = delete;
  ~Case() { pmpp::release(X); pmpp::release(Wq); pmpp::release(Wd); pmpp::release(Y); }

  // Y = NaN, guard band = sentinel.
  void refill() {
    std::vector<float> v((size_t)batch * out + guard, std::numeric_limits<float>::quiet_NaN());
    std::fill(v.begin() + (size_t)batch * out, v.end(), kSentinel);
    pmpp::upload(Y, v.data(), v.size() * 4);
  }

  // What the data must contain for the case to mean what the README says.
  std::string dataProblem() const {
    const bool hi = std::find(q.begin(), q.end(), (int8_t)127) != q.end();
    const bool lo = std::find(q.begin(), q.end(), (int8_t)-127) != q.end();
    if (!hi || !lo) return "the test data lost its codes 127 and -127 (a driver bug)";
    if (zeroRow >= 0)
      for (int i = 0; i < in / QK8_0; ++i)
        if (d[(size_t)zeroRow * (in / QK8_0) + i] != 0) return "the zero row has a nonzero scale (a driver bug)";
    return "";
  }

  // The reference depends only on the shape, so it is computed once per shape.
  void reference() {
    static std::map<std::tuple<int, int, int>, std::pair<std::vector<float>, std::vector<float>>> cache;
    auto& e = cache[std::make_tuple(batch, in, out)];
    if (e.first.empty()) {
      std::vector<float> deq((size_t)out * in);
      for (int o = 0; o < out; ++o)
        dequantizeRowQ8_0(q.data() + (size_t)o * in, d.data() + (size_t)o * (in / QK8_0), deq.data() + (size_t)o * in, in);
      cpuGemvDequant(x, deq, batch, in, out, e.first, e.second);
    }
    ref = &e.first;
    mag = &e.second;
  }

  // Empty if Y matches the reference and the guard band is untouched.
  std::string check() {
    if (!ref) reference();
    const size_t n = (size_t)batch * out;
    std::vector<float> got(n + guard);
    pmpp::download(Y, got.data(), got.size() * 4);
    std::string err = pmpp::describe(pmpp::compare(got.data(), ref->data(), mag->data(), n), out);
    size_t changed = 0, first = 0;
    for (size_t i = n; i < got.size(); ++i) {
      if (std::memcmp(&got[i], &kSentinel, 4) != 0) {
        if (changed == 0) first = i - n;
        ++changed;
      }
    }
    if (changed) {
      char b[300];
      std::snprintf(b, sizeof b,
                    "%s%zu of the %zu floats after Y were overwritten; the first, %zu floats past the end, is "
                    "where row %zu, column %zu of Y would be if Y had more rows",
                    err.empty() ? "" : "; ", changed, guard, first, (size_t)batch + first / out, first % out);
      err += b;
    }
    return err;
  }
};

static std::string shapeText(const Shape& s) {
  char b[200];
  std::snprintf(b, sizeof b, "batch %d, in %d, out %d  (%s)", s.batch, s.in, s.out, s.why);
  return b;
}

// ---------------------------------------------------------------- tests
static std::string hex4(unsigned v) { char b[16]; std::snprintf(b, sizeof b, "0x%04X", v); return b; }

// FP16: the hand-derived values, then every one of the 65536 bit patterns through
// halfBitsToFloat and back.
static std::string checkHalf() {
  for (const HalfFixture& f : kHalfFixtures) {
    const uint16_t got = floatToHalfBits(f.value);
    if (got != f.bits)
      return "floatToHalfBits(" + std::to_string(f.value) + ") gave " + hex4(got) + ", expected " + hex4(f.bits) +
             " (" + f.why + ")";
  }
  for (uint32_t h = 0; h < 65536; ++h) {
    const float v = halfBitsToFloat((uint16_t)h);
    const uint16_t back = floatToHalfBits(v);
    const bool nan = ((h >> 10) & 0x1F) == 0x1F && (h & 0x3FF);
    if (nan ? !std::isnan(v) : back != h)
      return "FP16 pattern " + hex4(h) + " decodes to " + std::to_string(v) + ", which encodes back as " + hex4(back);
  }
  return "";
}

static std::string checkQ8Fixture() {
  float x[QK8_0];
  for (int i = 0; i < QK8_0; ++i) x[i] = kQ8Multipliers[i] / 32.0f;
  int8_t q[QK8_0];
  uint16_t d;
  quantizeRowQ8_0Ref(x, q, &d, QK8_0);
  if (d != kQ8ScaleBits) return "scale bits " + hex4(d) + ", expected " + hex4(kQ8ScaleBits) + " (1/32)";
  for (int i = 0; i < QK8_0; ++i)
    if (q[i] != kQ8Codes[i])
      return "code " + std::to_string(i) + " (weight " + std::to_string(kQ8Multipliers[i]) + "/32) is " +
             std::to_string(q[i]) + ", expected " + std::to_string(kQ8Codes[i]);
  float zero[QK8_0] = {};
  quantizeRowQ8_0Ref(zero, q, &d, QK8_0);
  for (int i = 0; i < QK8_0; ++i)
    if (q[i] != 0) return "an all-zero block gave a nonzero code";
  if (d != 0) return "an all-zero block gave scale bits " + hex4(d) + ", expected 0x0000";
  return "";
}

static std::vector<Shape> shapes() {
  std::vector<Shape> s = {
      {1, 32, 1, "one block, one output"},
      {1, 64, 5, "two blocks; out not a multiple of the 4 warps of a block"},
      {2, 96, 7, "three blocks: fewer words than lanes in a row"},
      {3, 256, 64, "eight blocks"},
      {8, 512, 130, "the largest batch; out not a multiple of 4"},
      {5, 4096, 3, "a model-width row: several steps per lane"},
  };
  if (!pmpp::is_emu()) {
    s.push_back({1, 4096, 4096, "a 4096-wide layer at batch 1"});
    s.push_back({8, 4096, 4096, "a 4096-wide layer at the largest batch"});
    s.push_back({4, 14336, 4096, "an MLP down-projection; shape ours"});
    s.push_back({1, 4096, 14336, "an MLP up-projection; shape ours"});
    s.push_back({8, 1024, 4099, "out not a multiple of the 4 warps of a block"});
  }
  return s;
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("warp width: %d (%s)%s\n", warpWidth(), widthSource().c_str(),
              pmpp::is_emu() ? "; the emulators model 32-lane warps only" : "");

  // The driver's own parts first: if these fail, nothing below means anything.
  t.check("CPU: FP16 conversion on hand-derived values and all 65536 patterns", checkHalf);
  t.check("CPU: Q8_0 quantiser on the hand-derived fixture block and an all-zero block", checkQ8Fixture);

  for (const Shape& s : shapes()) {
    t.check("q8_0: " + shapeText(s), [&]() -> std::string {
      const std::string why = q8Problem();
      if (!why.empty()) return why;
      Case c(s);
      const std::string bad = c.dataProblem();
      if (!bad.empty()) return bad;
      c.refill();
      for (int launch = 0; launch < 2; ++launch) {
        runQ8(c.X, c.Wq, c.Wd, c.Y, s.batch, s.in, s.out);
        const std::string e = c.check();
        if (!e.empty()) return (launch ? "second launch on the same Y: " : "") + e;
      }
      return "";
    });
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// For each layer and batch, the F32 gather and your Q8_0 kernel take turns: one untimed
// round, then a.reps rounds, and each keeps its best round. One timed submission launches
// the kernel once per copy of its weights, going through copies that together hold at least
// 256 MB (at most 1024 copies), so a launch does not find its weights where the previous
// launch left them (as lab 22.1). The time reported is per launch.
static int bench(const pmpp::Args& a) {
  const std::string qp = q8Problem(), fp = f32Problem();
  if (!qp.empty() || !fp.empty()) {
    std::printf("bench: %s\n", (!qp.empty() ? qp : fp).c_str());
    return 1;
  }
  std::printf("warp width: %d (%s)\n", warpWidth(), widthSource().c_str());

  struct Layer { int in, out; const char* what; };
  const Layer layers[] = {
      {4096, 4096, "4096 -> 4096"},
      {4096, 14336, "4096 -> 14336, an MLP up-projection (shape ours)"},
      {14336, 4096, "14336 -> 4096, an MLP down-projection (shape ours)"},
  };
  const int batches[] = {1, 4, 8};
  const double kRotateBytes = 256e6;
  const double budget = memoryBudget();

  // Copies of `bytes` that together hold kRotateBytes, within a quarter of the budget.
  auto copiesFor = [&](double bytes) {
    int c = (int)std::min(1024.0, std::ceil(kRotateBytes / bytes));
    if (budget > 0) c = (int)std::min<double>(c, std::floor(budget / 4 / bytes));
    return c;
  };

  for (const Layer& L : layers) {
    const double f32Bytes = 4.0 * L.in * L.out;
    const double q8Bytes = (double)L.in * L.out + 2.0 * L.out * (L.in / QK8_0);   // codes + scales
    const int nF = copiesFor(f32Bytes), nQ = copiesFor(q8Bytes);
    if (nF < 1 || nQ < 1) {
      std::printf("\nlayer %s: skipped, one F32 copy of W (%.0f MB) is more than a quarter of the %.0f MB the device "
                  "reports\n", L.what, f32Bytes / 1e6, budget / 1e6);
      continue;
    }
    const std::vector<float> hw = pmpp::random_floats((size_t)L.in * L.out, 9200u + (uint32_t)L.out);
    const std::vector<float> hx = pmpp::random_floats((size_t)kMaxBatch * L.in, 9201u);
    std::vector<int8_t> hq;
    std::vector<uint16_t> hd;
    quantizeQ8_0(hw, L.in, L.out, hq, hd);

    std::vector<Buf*> Wf(nF), Wq(nQ), Wd(nQ);
    for (Buf*& b : Wf) { b = pmpp::alloc(hw.size() * 4); pmpp::upload(b, hw.data(), hw.size() * 4); }
    for (int i = 0; i < nQ; ++i) {
      Wq[i] = pmpp::alloc(hq.size());
      pmpp::upload(Wq[i], hq.data(), hq.size());
      Wd[i] = pmpp::alloc(hd.size() * 2);
      pmpp::upload(Wd[i], hd.data(), hd.size() * 2);
    }
    Buf* X = pmpp::alloc(hx.size() * 4);
    pmpp::upload(X, hx.data(), hx.size() * 4);
    Buf* Y = pmpp::alloc((size_t)kMaxBatch * L.out * 4);

    std::printf("\nlayer %s\n  F32 W %.1f MB x %d copies; Q8_0 W %.1f MB (codes and scales) x %d copies\n", L.what,
                f32Bytes / 1e6, nF, q8Bytes / 1e6, nQ);
    for (int batch : batches) {
      double best[2] = {std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
      auto once = [&](int k) -> double {
        const pmpp::Timing tm = pmpp::run([&](void* p) {
          if (k == 0) for (Buf* w : Wf) enqueueF32(p, X, w, Y, batch, L.in, L.out);
          else for (int i = 0; i < nQ; ++i) enqueueQ8(p, X, Wq[i], Wd[i], Y, batch, L.in, L.out);
        });
        return tm.best() / (k == 0 ? nF : nQ);
      };
      once(0); once(1);   // untimed round
      for (int r = 0; r < a.reps; ++r)
        for (int k = 0; k < 2; ++k) best[k] = std::min(best[k], once(k));
      const std::string err = pmpp::take_error();
      if (!err.empty()) {
        std::printf("  batch %d: a launch failed: %s\n", batch, err.c_str());
        continue;
      }
      std::printf("  batch %d   %-8s %-6s %12s %12s %12s %10s\n", batch, "kernel", "format", "time (us)", "W MB",
                  "W GB/s", "vs F32");
      const char* kernels[2] = {"gather", "q8_0"};
      const char* formats[2] = {"f32", "q8_0"};
      const double bytes[2] = {f32Bytes, q8Bytes};
      for (int k = 0; k < 2; ++k) {
        const double gbs = bytes[k] / best[k] / 1e9;
        std::printf("            %-8s %-6s %12.2f %12.1f %12.1f %9.2fx\n", kernels[k], formats[k], best[k] * 1e6,
                    bytes[k] / 1e6, gbs, best[0] / best[k]);
        pmpp::bench_row(a, "qgemv").s("kernel", kernels[k]).s("format", formats[k]).n("batch", batch).n("in", L.in)
            .n("out", L.out).n("seconds", best[k]).n("weight_bytes", bytes[k]).n("weight_gb_per_s", gbs)
            .n("speedup_vs_f32", best[0] / best[k]).emit();
      }
    }
    for (Buf* b : Wf) pmpp::release(b);
    for (int i = 0; i < nQ; ++i) { pmpp::release(Wq[i]); pmpp::release(Wd[i]); }
    pmpp::release(X);
    pmpp::release(Y);
  }
  std::printf("\n`W GB/s` counts the bytes of weights each format stores (Q8_0: codes and scales), divided by the\n"
              "time: not everything the memory system moved. These are measurements of this device at these\n"
              "shapes, not properties of the kernels.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
