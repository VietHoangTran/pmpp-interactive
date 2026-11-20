// Lab X2.2 driver: checks the CPU Q4_0 quantiser on hand-derived fixtures, tests your Q4_0
// matrix-vector product (and the provided unfused baseline) against a CPU reference, then
// times F32, Q8_0, unfused Q4_0 and your fused Q4_0 on decode-sized layers.
//   Y[b][j] = sum over k of X[b][k] * w[j][k],  w = (code - 8) * d,  1 <= batch <= 8, in % 32 == 0,
//   W [out x in/32] llama.cpp block_q4_0 (labs/x2/shared/qformat.h), X and Y float32, batch first.
// You do not need to edit this file; your work goes in work/.
//
// What a test case does (README, "What the tests do"):
//   - Each 32-weight block is scaled by its own power of two between 2^-12 and 2^12, holds a
//     +peak and a -peak (so one of them clips to code 15), and alternates which comes first
//     (so d is negative in some blocks and positive in others). One block of each row has
//     weights j and j + 16 of opposite signs, so pairing neighbours in a byte fails. Row
//     out / 2 (out >= 3) and the last block of row 0 (two blocks or more) are all zero:
//     d = 0, codes 8.
//   - X and W are allocated with exactly the bytes they hold; Y starts as NaN, is followed by
//     a guard band of sentinels, and each case launches twice without resetting Y.
//   - Results are compared with y = x dequant(W)^T in double, within 1e-4 x (1 + sum |x w|).
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

#if defined(PMPP_HAVE_Q8)
static const bool kHaveQ8 = true;
#else
static const bool kHaveQ8 = false;
#endif

// ---------------------------------------------------------------- the CPU reference
// Y[b][o] = sum_k X[b][k] * w[o][k], w already dequantised, accumulated in double;
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

static void quantizeQ4_0(const std::vector<float>& w, int in, int out, std::vector<BlockQ4_0>& q) {
  q.resize((size_t)out * (in / QK4_0));
  for (int o = 0; o < out; ++o) quantizeRowQ4_0Ref(w.data() + (size_t)o * in, q.data() + (size_t)o * (in / QK4_0), in);
}
static void quantizeQ8_0(const std::vector<float>& w, int in, int out, std::vector<int8_t>& q, std::vector<uint16_t>& d) {
  q.resize((size_t)out * in);
  d.resize((size_t)out * (in / QK8_0));
  for (int o = 0; o < out; ++o)
    quantizeRowQ8_0Ref(w.data() + (size_t)o * in, q.data() + (size_t)o * in, d.data() + (size_t)o * (in / QK8_0), in);
}

// ---------------------------------------------------------------- backends
// Each backend section defines:
//   q4Problem(), unfusedProblem(), f32Problem(), q8Problem()   empty if that kernel can run here
//   enqueueQ4(pass, X, W, Y, ...)                 one launch of your kernel
//   enqueueUnfused(pass, X, W, Wf, Y, ...)        the provided dequantise-then-gather path
//   enqueueF32(...), enqueueQ8(...)               lab 22.1's gather, lab X2.1's Q8_0 kernel
//   warpWidth(), widthSource(), memoryBudget()
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
void encodeGemvQ4_0(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                    id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvQ4_0Unfused(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState> dequantPipeline,
                           id<MTLComputePipelineState> gatherPipeline, id<MTLBuffer> X, id<MTLBuffer> W,
                           id<MTLBuffer> Wf, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvGather(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                      id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
#if defined(PMPP_HAVE_Q8)
void encodeGemvQ8_0(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> Wq,
                    id<MTLBuffer> Wd, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
#endif
using pmpp::metal::buffer;

static std::string dirOf(const std::string& p) { return p.substr(0, p.find_last_of('/')); }
// PMPP_METAL_SOURCE is labs/x2/lab2_q4gemv/<variant>/kernel.metal.
static const std::string kLab = dirOf(dirOf(PMPP_METAL_SOURCE));
static const std::string kLabs = dirOf(dirOf(kLab));
static const std::string kGatherSource = kLabs + "/ch22/lab1_gemv/solution/kernel.metal";
static const std::string kQ8Source = kLabs + "/x2/lab1_q8gemv/solution/kernel.metal";
static const std::string kUnfusedSource = kLab + "/provided/unfused.metal";

static id<MTLComputePipelineState> q4Pso() { return pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gemv_q4_0"); }
static id<MTLComputePipelineState> f32Pso() { return pmpp::metal::pipeline(kGatherSource, "gemv_gather"); }
static id<MTLComputePipelineState> q8Pso() { return pmpp::metal::pipeline(kQ8Source, "gemv_q8_0"); }
static id<MTLComputePipelineState> dqPso() { return pmpp::metal::pipeline(kUnfusedSource, "dequant_q4_0"); }
static std::string pipelineProblem(id<MTLComputePipelineState> p, const std::string& what) {
  if (!p) return what + " did not build: " + pmpp::take_error();
  if (p.threadExecutionWidth == 0 || p.maxTotalThreadsPerThreadgroup < p.threadExecutionWidth)
    return what + ": this pipeline cannot run one whole SIMD-group per threadgroup";
  return "";
}
static std::string q4Problem() {
  std::string e = pipelineProblem(q4Pso(), "gemv_q4_0 in your kernel.metal");
  if (e.empty() && q4Pso().threadExecutionWidth % 4 != 0)
    e = "the SIMD-group width is not a multiple of the 4 lanes that share a block";
  return e;
}
static std::string f32Problem() { return pipelineProblem(f32Pso(), "gemv_gather in " + kGatherSource); }
static std::string q8Problem() { return kHaveQ8 ? pipelineProblem(q8Pso(), "gemv_q8_0 in " + kQ8Source) : "lab X2.1's solution is not present"; }
static std::string unfusedProblem() {
  std::string e = pipelineProblem(dqPso(), "dequant_q4_0 in " + kUnfusedSource);
  return e.empty() ? f32Problem() : e;
}
static id<MTLComputeCommandEncoder> enc(void* vp) { return ((pmpp::metal::Pass*)vp)->encoder; }
static void enqueueQ4(void* vp, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  if (id<MTLComputePipelineState> p = q4Pso())
    encodeGemvQ4_0(enc(vp), p, buffer(X), buffer(W), buffer(Y), (uint32_t)batch, (uint32_t)in, (uint32_t)out);
}
static void enqueueUnfused(void* vp, Buf* X, Buf* W, Buf* Wf, Buf* Y, int batch, int in, int out) {
  id<MTLComputePipelineState> d = dqPso(), g = f32Pso();
  if (d && g)
    encodeGemvQ4_0Unfused(enc(vp), d, g, buffer(X), buffer(W), buffer(Wf), buffer(Y), (uint32_t)batch, (uint32_t)in,
                          (uint32_t)out);
}
static void enqueueF32(void* vp, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  if (id<MTLComputePipelineState> p = f32Pso())
    encodeGemvGather(enc(vp), p, buffer(X), buffer(W), buffer(Y), (uint32_t)batch, (uint32_t)in, (uint32_t)out);
}
static void enqueueQ8(void* vp, Buf* X, Buf* Wq, Buf* Wd, Buf* Y, int batch, int in, int out) {
#if defined(PMPP_HAVE_Q8)
  if (id<MTLComputePipelineState> p = q8Pso())
    encodeGemvQ8_0(enc(vp), p, buffer(X), buffer(Wq), buffer(Wd), buffer(Y), (uint32_t)batch, (uint32_t)in,
                   (uint32_t)out);
#else
  (void)vp; (void)X; (void)Wq; (void)Wd; (void)Y; (void)batch; (void)in; (void)out;
#endif
}
// Apple documents no fixed SIMD-group width; the pipeline reports it (ch03-facts C9).
static int warpWidth() {
  id<MTLComputePipelineState> p = q4Pso();
  if (!p) p = f32Pso();
  return p ? (int)p.threadExecutionWidth : 0;
}
static std::string widthSource() { return "threadExecutionWidth of the gemv_q4_0 pipeline"; }
static double memoryBudget() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }

#else  // CUDA, HIP and the emulators
void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out);
void launchGemvQ4_0Unfused(const float* X, const BlockQ4_0* W, float* Wf, float* Y, int batch, int in, int out);
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 22.1
#if defined(PMPP_HAVE_Q8)
void launchGemvQ8_0(const float* X, const int8_t* Wq, const uint16_t* Wd, float* Y, int batch, int in, int out);
#endif

// Blocks of 4 warps (WARPS_PER_BLOCK in kernel.cu); 4 lanes share a Q4_0 block.
static std::string blockProblem() {
  const int limit = (int)pmpp::device_prop("maxThreadsPerBlock", 1024);
  const int width = (int)pmpp::device_prop("warpSize", 0);
  if (width <= 0) return "the device reports no warpSize";
  if (width % 4 != 0) return "the warp width is not a multiple of the 4 lanes that share a block";
  if (limit < 4 * width) return "this device allows " + std::to_string(limit) + " threads per block; the kernels need 4 warps";
  return "";
}
static std::string q4Problem() { return blockProblem(); }
static std::string unfusedProblem() { return blockProblem(); }
static std::string f32Problem() { return blockProblem(); }
static std::string q8Problem() { return kHaveQ8 ? blockProblem() : "lab X2.1's solution is not present"; }
static const float* cf(Buf* b) { return (const float*)pmpp::raw(b); }
static float* mf(Buf* b) { return (float*)pmpp::raw(b); }
static void enqueueQ4(void*, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  launchGemvQ4_0(cf(X), (const BlockQ4_0*)pmpp::raw(W), mf(Y), batch, in, out);
}
static void enqueueUnfused(void*, Buf* X, Buf* W, Buf* Wf, Buf* Y, int batch, int in, int out) {
  launchGemvQ4_0Unfused(cf(X), (const BlockQ4_0*)pmpp::raw(W), mf(Wf), mf(Y), batch, in, out);
}
static void enqueueF32(void*, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  launchGemvGather(cf(X), cf(W), mf(Y), batch, in, out);
}
static void enqueueQ8(void*, Buf* X, Buf* Wq, Buf* Wd, Buf* Y, int batch, int in, int out) {
#if defined(PMPP_HAVE_Q8)
  launchGemvQ8_0(cf(X), (const int8_t*)pmpp::raw(Wq), (const uint16_t*)pmpp::raw(Wd), mf(Y), batch, in, out);
#else
  (void)X; (void)Wq; (void)Wd; (void)Y; (void)batch; (void)in; (void)out;
#endif
}
static int warpWidth() { return (int)pmpp::device_prop("warpSize", 0); }
static std::string widthSource() { return "warpSize from the device properties"; }
static double memoryBudget() { return pmpp::device_prop("totalGlobalMem", 0); }
#endif

// ---------------------------------------------------------------- test weights
// Uniform values in [-1, 1], then for block i of row j: times s = 2^e with
// e = ((3j + 5i) mod 25) - 12, from 2^-12 to 2^12 (neighbouring blocks differ by 2^5).
// Weights (i mod 32) and ((i + 16) mod 32) are set to +s and -s, the order alternating with
// i + j: Q4_0 takes the first of two equal magnitudes as its signed extreme, so d is
// negative in half the blocks and positive in the other half, and the other extreme clips
// to code 15. In block (j mod blocks) the first 16 weights are positive and the last 16
// negative, so a byte read as two neighbours instead of j and j + 16 gives a wrong answer.
// Row out / 2 (out >= 3) and the last block of row 0 (two blocks or more) are all zero.
static std::vector<float> testWeights(int in, int out, uint32_t seed, int* zeroRow, int* zeroBlock) {
  std::vector<float> w = pmpp::random_floats((size_t)out * in, seed);
  const int nb = in / QK4_0;
  *zeroRow = out >= 3 ? out / 2 : -1;
  *zeroBlock = nb >= 2 ? nb - 1 : -1;
  for (int j = 0; j < out; ++j) {
    for (int i = 0; i < nb; ++i) {
      float* p = w.data() + (size_t)j * in + (size_t)i * QK4_0;
      const float s = std::ldexp(1.0f, ((3 * j + 5 * i) % 25) - 12);
      if (i == j % nb)
        for (int k = 0; k < QK4_0; ++k) p[k] = std::fabs(p[k]) * (k < QK4_0 / 2 ? 1.0f : -1.0f);
      for (int k = 0; k < QK4_0; ++k) p[k] *= s;
      const float first = ((i + j) % 2 == 0) ? s : -s;
      p[std::min(i % QK4_0, (i + 16) % QK4_0)] = first;
      p[std::max(i % QK4_0, (i + 16) % QK4_0)] = -first;
      if (j == *zeroRow || (j == 0 && i == *zeroBlock)) std::fill(p, p + QK4_0, 0.0f);
    }
  }
  return w;
}

// ---------------------------------------------------------------- a test problem on the device
static const float kSentinel = -77777.0f;   // no correct kernel writes anything past Y

struct Shape { int batch, in, out; const char* why; };

struct Case {
  int batch, in, out, zeroRow = -1, zeroBlock = -1;
  size_t guard;
  std::vector<float> x, w;
  std::vector<BlockQ4_0> q;
  const std::vector<float>* ref = nullptr;
  const std::vector<float>* mag = nullptr;
  Buf *X = nullptr, *W = nullptr, *Y = nullptr, *Wf = nullptr;

  Case(const Shape& s, bool scratch) : batch(s.batch), in(s.in), out(s.out) {
    guard = pmpp::is_emu() ? 256 : (size_t)kMaxBatch * out + 256;
    const uint32_t seed = 94000u + (uint32_t)batch * 131u + (uint32_t)in * 17u + (uint32_t)out;
    x = pmpp::random_floats((size_t)batch * in, seed);
    w = testWeights(in, out, seed + 1, &zeroRow, &zeroBlock);
    quantizeQ4_0(w, in, out, q);
    X = pmpp::alloc(x.size() * 4);
    W = pmpp::alloc(q.size() * sizeof(BlockQ4_0));   // exactly 18 bytes per block
    Y = pmpp::alloc(((size_t)batch * out + guard) * 4);
    if (scratch) Wf = pmpp::alloc((size_t)out * in * 4);
    pmpp::upload(X, x.data(), x.size() * 4);
    pmpp::upload(W, q.data(), q.size() * sizeof(BlockQ4_0));
  }
  Case(const Case&) = delete;
  ~Case() { pmpp::release(X); pmpp::release(W); pmpp::release(Y); if (Wf) pmpp::release(Wf); }

  void refill() {
    std::vector<float> v((size_t)batch * out + guard, std::numeric_limits<float>::quiet_NaN());
    std::fill(v.begin() + (size_t)batch * out, v.end(), kSentinel);
    pmpp::upload(Y, v.data(), v.size() * 4);
  }

  // What the data must contain for the case to mean what the README says.
  std::string dataProblem() const {
    bool neg = false, pos = false, fifteen = false;
    for (const BlockQ4_0& b : q) {
      if ((b.d & 0x7FFFu) == 0) {}          // d = +0 or -0: a zero block
      else if (b.d & 0x8000u) neg = true;
      else pos = true;
      for (uint8_t v : b.qs) if ((v & 0x0F) == 15 || (v >> 4) == 15) fifteen = true;
    }
    if (q.size() >= 2 && !(neg && pos)) return "the test data lacks a block of each sign of d (a driver bug)";
    if (!fifteen) return "the test data has no code 15 (a driver bug)";
    const int nb = in / QK4_0;
    auto zero = [&](size_t i) {
      if ((q[i].d & 0x7FFFu) != 0) return false;   // 0 / -8 gives -0: 0x8000 (qfixture.h)
      for (uint8_t v : q[i].qs) if (v != 0x88) return false;
      return true;
    };
    if (zeroRow >= 0)
      for (int i = 0; i < nb; ++i) if (!zero((size_t)zeroRow * nb + i)) return "the zero row is not d = 0, codes 8 (a driver bug)";
    if (zeroBlock >= 0 && !zero((size_t)zeroBlock)) return "the zero block is not d = 0, codes 8 (a driver bug)";
    return "";
  }

  void reference() {
    static std::map<std::tuple<int, int, int>, std::pair<std::vector<float>, std::vector<float>>> cache;
    auto& e = cache[std::make_tuple(batch, in, out)];
    if (e.first.empty()) {
      std::vector<float> deq((size_t)out * in);
      for (int o = 0; o < out; ++o)
        dequantizeRowQ4_0(q.data() + (size_t)o * (in / QK4_0), deq.data() + (size_t)o * in, in);
      cpuGemvDequant(x, deq, batch, in, out, e.first, e.second);
    }
    ref = &e.first;
    mag = &e.second;
  }

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

// One fixture block: quantise `w`, compare the scale bits and the packed bytes, then check
// that dequantising gives (code - 8) * d.
static std::string checkQ4Block(const float* w, const uint8_t* codes, uint16_t dbits, const char* name) {
  BlockQ4_0 b;
  quantizeRowQ4_0Ref(w, &b, QK4_0);
  if (b.d != dbits) return std::string(name) + ": scale bits " + hex4(b.d) + ", expected " + hex4(dbits);
  for (int j = 0; j < QK4_0 / 2; ++j) {
    const uint8_t want = (uint8_t)(codes[j] | (codes[j + 16] << 4));
    if (b.qs[j] != want) {
      char m[200];
      std::snprintf(m, sizeof m, "%s: byte %d is 0x%02X, expected 0x%02X (code %d low, code %d high)", name, j,
                    b.qs[j], want, j, j + 16);
      return m;
    }
  }
  float y[QK4_0];
  dequantizeRowQ4_0(&b, y, QK4_0);
  const float d = halfBitsToFloat(dbits);
  for (int k = 0; k < QK4_0; ++k)
    if (y[k] != (codes[k] - 8) * d)
      return std::string(name) + ": weight " + std::to_string(k) + " dequantises to " + std::to_string(y[k]);
  return "";
}

static std::string checkQ4Fixtures() {
  float w[QK4_0];
  for (int i = 0; i < QK4_0; ++i) w[i] = kQ4aU[i] / 8.0f;
  std::string e = checkQ4Block(w, kQ4aCodes, kQ4aScaleBits, "negative-extreme block");
  if (!e.empty()) return e;
  e = checkQ4Block(kQ4bW, kQ4bCodes, kQ4bScaleBits, "positive-extreme block");
  if (!e.empty()) return e;
  const float zero[QK4_0] = {};
  uint8_t eights[QK4_0];
  std::fill(eights, eights + QK4_0, (uint8_t)8);
  return checkQ4Block(zero, eights, kQ4ZeroScaleBits, "all-zero block (d = 0 / -8 is negative zero)");
}

static std::vector<Shape> shapes() {
  std::vector<Shape> s = {
      {1, 32, 1, "one block, one output"},
      {1, 64, 5, "two blocks; out not a multiple of the 4 warps of a block"},
      {2, 96, 7, "three blocks: an odd number"},
      {3, 256, 64, "eight blocks"},
      {8, 512, 130, "the largest batch; out not a multiple of 4"},
      {5, 4096, 3, "a model-width row: several steps per lane"},
      {1, 32 * 33, 3, "33 blocks per row: every other row starts at 2 mod 4"},
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

static std::string runCase(const Shape& s, bool unfused) {
  const std::string why = unfused ? unfusedProblem() : q4Problem();
  if (!why.empty()) return why;
  Case c(s, unfused);
  const std::string bad = c.dataProblem();
  if (!bad.empty()) return bad;
  c.refill();
  for (int launch = 0; launch < 2; ++launch) {
    pmpp::run([&](void* p) {
      if (unfused) enqueueUnfused(p, c.X, c.W, c.Wf, c.Y, s.batch, s.in, s.out);
      else enqueueQ4(p, c.X, c.W, c.Y, s.batch, s.in, s.out);
    });
    const std::string e = c.check();
    if (!e.empty()) return (launch ? "second launch on the same Y: " : "") + e;
  }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("warp width: %d (%s)%s\n", warpWidth(), widthSource().c_str(),
              pmpp::is_emu() ? "; the emulators model 32-lane warps only" : "");

  // The driver's own parts first: if these fail, nothing below means anything.
  t.check("CPU: Q4_0 quantiser on the hand-derived fixture blocks (both signs of d) and an all-zero block",
          checkQ4Fixtures);

  for (const Shape& s : shapes()) t.check("q4_0: " + shapeText(s), [&]() { return runCase(s, false); });
  // The provided baseline, so that the benchmark compares correct kernels. Three shapes.
  for (const Shape& s : shapes())
    if (s.in == 32 * 33 || s.out == 130 || s.out == 4099)
      t.check("provided unfused q4_0: " + shapeText(s), [&]() { return runCase(s, true); });
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// For each layer and batch, F32 (lab 22.1's gather), Q8_0 (lab X2.1's solution, if present),
// unfused Q4_0 (provided) and your Q4_0 take turns: one untimed round, then a.reps rounds,
// each keeping its best. One timed submission launches the kernel once per copy of its
// weights, going through copies that together hold at least 256 MB per format (at most 1024
// copies), as lab 22.1. The unfused path writes and reads one scratch matrix of out x in
// floats; it is not copied.
static int bench(const pmpp::Args& a) {
  for (const std::string& e : {q4Problem(), unfusedProblem(), f32Problem()})
    if (!e.empty()) { std::printf("bench: %s\n", e.c_str()); return 1; }
  const bool q8 = q8Problem().empty();
  if (!q8) std::printf("no Q8_0 rows: %s\n", q8Problem().c_str());
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
  auto copiesFor = [&](double bytes) {
    int c = (int)std::min(1024.0, std::ceil(kRotateBytes / bytes));
    if (budget > 0) c = (int)std::min<double>(c, std::floor(budget / 5 / bytes));
    return c;
  };

  enum { F32, Q8, UNF, Q4, N };
  const char* kernels[N] = {"gather", "q8_0", "q4_0_unfused", "q4_0"};
  const char* formats[N] = {"f32", "q8_0", "q4_0", "q4_0"};

  for (const Layer& L : layers) {
    const double nw = (double)L.in * L.out;
    const double bytes[N] = {4.0 * nw, nw + 2.0 * L.out * (L.in / QK8_0), 18.0 * nw / QK4_0, 18.0 * nw / QK4_0};
    const int nF = copiesFor(bytes[F32]), nQ8 = copiesFor(bytes[Q8]), nQ4 = copiesFor(bytes[Q4]);
    if (nF < 1 || nQ8 < 1 || nQ4 < 1 || (budget > 0 && bytes[F32] > budget / 5)) {
      std::printf("\nlayer %s: skipped, one F32 copy of W (%.0f MB) is more than a fifth of the %.0f MB the device "
                  "reports\n", L.what, bytes[F32] / 1e6, budget / 1e6);
      continue;
    }
    const std::vector<float> hw = pmpp::random_floats((size_t)nw, 9400u + (uint32_t)L.out);
    const std::vector<float> hx = pmpp::random_floats((size_t)kMaxBatch * L.in, 9401u);
    std::vector<BlockQ4_0> h4;
    quantizeQ4_0(hw, L.in, L.out, h4);
    std::vector<int8_t> hq;
    std::vector<uint16_t> hd;
    if (q8) quantizeQ8_0(hw, L.in, L.out, hq, hd);

    std::vector<Buf*> Wf32(nF), W4(nQ4), Wq(q8 ? nQ8 : 0), Wd(q8 ? nQ8 : 0);
    for (Buf*& b : Wf32) { b = pmpp::alloc(hw.size() * 4); pmpp::upload(b, hw.data(), hw.size() * 4); }
    for (Buf*& b : W4) { b = pmpp::alloc(h4.size() * sizeof(BlockQ4_0)); pmpp::upload(b, h4.data(), h4.size() * sizeof(BlockQ4_0)); }
    for (size_t i = 0; i < Wq.size(); ++i) {
      Wq[i] = pmpp::alloc(hq.size()); pmpp::upload(Wq[i], hq.data(), hq.size());
      Wd[i] = pmpp::alloc(hd.size() * 2); pmpp::upload(Wd[i], hd.data(), hd.size() * 2);
    }
    Buf* scratch = pmpp::alloc((size_t)nw * 4);
    Buf* X = pmpp::alloc(hx.size() * 4);
    pmpp::upload(X, hx.data(), hx.size() * 4);
    Buf* Y = pmpp::alloc((size_t)kMaxBatch * L.out * 4);

    std::printf("\nlayer %s\n  F32 W %.1f MB x %d copies; Q8_0 %.1f MB x %d; Q4_0 %.1f MB x %d; unfused scratch %.1f MB\n",
                L.what, bytes[F32] / 1e6, nF, bytes[Q8] / 1e6, q8 ? nQ8 : 0, bytes[Q4] / 1e6, nQ4, bytes[F32] / 1e6);
    for (int batch : batches) {
      double best[N];
      std::fill(best, best + N, std::numeric_limits<double>::infinity());
      auto once = [&](int k) -> double {
        const pmpp::Timing tm = pmpp::run([&](void* p) {
          if (k == F32) for (Buf* w : Wf32) enqueueF32(p, X, w, Y, batch, L.in, L.out);
          if (k == Q8) for (size_t i = 0; i < Wq.size(); ++i) enqueueQ8(p, X, Wq[i], Wd[i], Y, batch, L.in, L.out);
          if (k == UNF) for (Buf* w : W4) enqueueUnfused(p, X, w, scratch, Y, batch, L.in, L.out);
          if (k == Q4) for (Buf* w : W4) enqueueQ4(p, X, w, Y, batch, L.in, L.out);
        });
        const int n = k == F32 ? nF : k == Q8 ? nQ8 : nQ4;
        return tm.best() / n;
      };
      for (int k = 0; k < N; ++k) if (k != Q8 || q8) once(k);   // untimed round
      for (int r = 0; r < a.reps; ++r)
        for (int k = 0; k < N; ++k) if (k != Q8 || q8) best[k] = std::min(best[k], once(k));
      const std::string err = pmpp::take_error();
      if (!err.empty()) {
        std::printf("  batch %d: a launch failed: %s\n", batch, err.c_str());
        continue;
      }
      std::printf("  batch %d   %-13s %-6s %12s %10s %10s %9s %9s\n", batch, "kernel", "format", "time (us)", "W MB",
                  "W GB/s", "vs F32", "vs Q8_0");
      for (int k = 0; k < N; ++k) {
        if (k == Q8 && !q8) continue;
        const double gbs = bytes[k] / best[k] / 1e9;
        char vq8[32] = "-";
        if (q8) std::snprintf(vq8, sizeof vq8, "%.2fx", best[Q8] / best[k]);
        std::printf("            %-13s %-6s %12.2f %10.1f %10.1f %8.2fx %9s\n", kernels[k], formats[k], best[k] * 1e6,
                    bytes[k] / 1e6, gbs, best[F32] / best[k], vq8);
        pmpp::bench_row(a, "qgemv").s("kernel", kernels[k]).s("format", formats[k]).n("batch", batch).n("in", L.in)
            .n("out", L.out).n("seconds", best[k]).n("weight_bytes", bytes[k]).n("weight_gb_per_s", gbs)
            .n("speedup_vs_f32", best[F32] / best[k]).emit();
      }
    }
    for (Buf* b : Wf32) pmpp::release(b);
    for (Buf* b : W4) pmpp::release(b);
    for (size_t i = 0; i < Wq.size(); ++i) { pmpp::release(Wq[i]); pmpp::release(Wd[i]); }
    pmpp::release(scratch);
    pmpp::release(X);
    pmpp::release(Y);
  }
  std::printf("\n`W GB/s` counts the bytes of weights each format stores (Q8_0: codes and scales; Q4_0: 18-byte\n"
              "blocks), divided by the time. The unfused path also writes and reads a float32 copy of W, which\n"
              "this figure does not count. These are measurements of this device at these shapes, not\n"
              "properties of the kernels.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
