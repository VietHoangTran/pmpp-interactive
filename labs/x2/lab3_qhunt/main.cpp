// Lab X2.3 driver: tests the Q4_0 x Q8_1 integer-path GEMV (which ships with four defects)
// against a CPU reference of the same integer path, then measures the integer path against
// the float path of lab X2.2. You do not need to edit this file; your work goes in work/.
//
//   qhunt.h                     the functions kernel.cu defines
//   ../shared-g12/qformats.h    the block formats and the CPU quantisers (llama.cpp's rules)
//   ../shared-g12/qpieces.cmake where the float Q4_0 GEMV comes from: lab X2.2's solution, or
//                               a stand-in (the driver prints which)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "qhunt.h"
#include "pmpp/harness.hpp"
using pmpp::Buf;

#ifndef PMPP_Q4_0_FROM
#define PMPP_Q4_0_FROM "unknown"
#endif

// ---------------------------------------------------------------- backends
// enq* put one operation into the current pass (a Metal encoder, or the default stream);
// exec runs a pass and times it on the device.
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
using pmpp::metal::buffer;
bool encodeQuantizeQ8_1(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> Xq,
                        uint32_t batch, uint32_t in);
void encodeGemvQ4_0Q8_1(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> W, id<MTLBuffer> Xq,
                        id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);
void encodeGemvQ4_0(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                    id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);   // lab X2.2's, or the stand-in

struct Pipes {
  id<MTLComputePipelineState> quant = nil, gemv = nil, flt = nil;
  std::string problem;
};
static Pipes& pipes() {
  static Pipes p;
  static bool done = false;
  if (!done) {
    done = true;
    p.quant = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "quantize_q8_1");
    if (!p.quant) p.problem = "kernel.metal's quantize_q8_1 did not build: " + pmpp::take_error();
    p.gemv = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "gemv_q4_0_q8_1");
    if (!p.gemv && p.problem.empty()) p.problem = "kernel.metal's gemv_q4_0_q8_1 did not build: " + pmpp::take_error();
    p.flt = pmpp::metal::pipeline(PMPP_Q4_0_METAL, PMPP_Q4_0_KERNEL);
    if (p.quant && p.quant.threadExecutionWidth % 32 != 0 && p.problem.empty())
      p.problem = "this pipeline's SIMD-group width is not a multiple of 32; the provided quantiser needs that";
  }
  return p;
}
static id<MTLComputeCommandEncoder> enc(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }
static bool enqQuantize(void* pass, Buf* X, Buf* Xq, int batch, int in) {
  return encodeQuantizeQ8_1(enc(pass), pipes().quant, buffer(X), buffer(Xq), batch, in);
}
static void enqGemv(void* pass, Buf* W, Buf* Xq, Buf* Y, int batch, int in, int out) {
  encodeGemvQ4_0Q8_1(enc(pass), pipes().gemv, buffer(W), buffer(Xq), buffer(Y), batch, in, out);
}
static void enqFloat(void* pass, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  encodeGemvQ4_0(enc(pass), pipes().flt, buffer(X), buffer(W), buffer(Y), batch, in, out);
}
static std::string setupProblem() { return pipes().problem; }
static std::string floatProblem() {
  return pipes().flt ? "" : std::string("the float Q4_0 GEMV did not build: ") + pmpp::take_error();
}
static pmpp::Timing exec(const std::function<void(void*)>& f) { return pmpp::run(f); }
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* timingUnit() { return "one command buffer per run, GPUEndTime - GPUStartTime"; }
#else
void launchGemvQ4_0(const float* X, const BlockQ4_0* W, float* Y, int batch, int in, int out);   // X2.2's, or the stand-in
float eventTimeMs(void (*enqueue)(void* ctx), void* ctx);   // labs/ch22/lab2_batching/provided/step_timer.cu
template <class T> static T* p(Buf* b) { return (T*)pmpp::raw(b); }
static bool enqQuantize(void*, Buf* X, Buf* Xq, int batch, int in) {
  launchQuantizeQ8_1(p<float>(X), p<BlockQ8_1>(Xq), batch, in);
  return true;
}
static void enqGemv(void*, Buf* W, Buf* Xq, Buf* Y, int batch, int in, int out) {
  launchGemvQ4_0Q8_1(p<BlockQ4_0>(W), p<BlockQ8_1>(Xq), p<float>(Y), batch, in, out);
}
static void enqFloat(void*, Buf* X, Buf* W, Buf* Y, int batch, int in, int out) {
  launchGemvQ4_0(p<float>(X), p<BlockQ4_0>(W), p<float>(Y), batch, in, out);
}
static std::string setupProblem() { return ""; }
static std::string floatProblem() { return ""; }
// One pair of events around the pass (the device's own time), inside pmpp::run, which waits
// and collects launch errors.
static pmpp::Timing exec(const std::function<void(void*)>& f) {
  float ms = -1.0f;
  pmpp::Timing t = pmpp::run([&](void*) {
    ms = eventTimeMs([](void* c) { (*(const std::function<void(void*)>*)c)(nullptr); }, (void*)&f);
  });
  if (ms >= 0) t.gpu_s = ms * 1e-3;
  return t;
}
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* timingUnit() { return "one pair of events per run"; }
#endif

// ---------------------------------------------------------------- buffers with guard bands
// Outputs are followed by kGuard bytes of kGuardByte and start as NaN (Y) or 0xEE bytes (Xq), so
// a value nobody wrote cannot pass and a write past the end shows. Inputs are allocated to
// their exact size: on `emu` a read past one is an AddressSanitizer error naming the line.
static const size_t kGuard = 256;
static const unsigned char kGuardByte = 0x5E;

static Buf* outputBuffer(size_t bytes, unsigned char fill, bool nanFloats) {
  Buf* b = pmpp::alloc(bytes + kGuard);
  std::vector<unsigned char> v(bytes + kGuard, kGuardByte);
  if (nanFloats) {
    const uint32_t nan = 0x7FC00000u;
    for (size_t i = 0; i + 4 <= bytes; i += 4) std::memcpy(&v[i], &nan, 4);
  } else {
    std::fill(v.begin(), v.begin() + bytes, fill);
  }
  pmpp::upload(b, v.data(), v.size());
  return b;
}
static bool guardIntact(const std::vector<unsigned char>& v, size_t bytes) {
  for (size_t i = bytes; i < v.size(); ++i)
    if (v[i] != kGuardByte) return false;
  return true;
}
static std::vector<unsigned char> fetch(Buf* b, size_t bytes) {
  std::vector<unsigned char> v(bytes + kGuard);
  pmpp::download(b, v.data(), v.size());
  return v;
}

// ---------------------------------------------------------------- test data
enum Kind { RANDOM, WORST, SPLIT16, POW2 };

static uint32_t xorshift(uint32_t& x) { x ^= x << 13; x ^= x >> 17; x ^= x << 5; return x; }

// Weights: (out, in) as out * in / 32 blocks.
static std::vector<BlockQ4_0> makeWeights(Kind kind, int out, int in, uint32_t seed) {
  const int nb = in / QK4_0;
  std::vector<BlockQ4_0> w((size_t)out * nb);
  uint32_t r = seed * 2654435761u + 7u;
  if (kind == RANDOM) {
    // Real weights through llama.cpp's _ref quantiser (qformats.h).
    std::vector<float> f = pmpp::random_floats((size_t)out * in, seed);
    for (float& x : f) x *= 0.05f;
    for (int o = 0; o < out; ++o) qref::quantizeQ4_0Row(&f[(size_t)o * in], &w[(size_t)o * nb], in);
    return w;
  }
  for (size_t k = 0; k < w.size(); ++k) {
    BlockQ4_0& b = w[k];
    if (kind == WORST) {
      b.d = qref::floatToHalf(0.01f);
      std::memset(b.qs, 0xFF, sizeof b.qs);              // every code 15: weight 7 d
    } else if (kind == SPLIT16) {
      // Weight j and weight j + 16 share byte j: give them opposite extremes, alternating by block.
      b.d = qref::floatToHalf(0.005f + 0.015f * (float)(xorshift(r) % 1000) / 1000.0f);
      std::memset(b.qs, (k % 2) ? 0xF0 : 0x0F, sizeof b.qs);
    } else {                                              // POW2: random codes, scale 2^(k mod 8 - 9)
      b.d = qref::floatToHalf(std::ldexp(1.0f, (int)(k % 8) - 9));
      for (int j = 0; j < 16; ++j) b.qs[j] = (uint8_t)(xorshift(r) & 0xFF);
    }
  }
  return w;
}

// Activations [batch x in]: uniform in [-1, 1], or for WORST a positive constant, so every
// Q8_1 code is 127 (x / d = amax / (amax / 127)).
static std::vector<float> makeActs(Kind kind, int batch, int in, uint32_t seed) {
  if (kind == WORST) return std::vector<float>((size_t)batch * in, 0.5f);
  return pmpp::random_floats((size_t)batch * in, seed);
}

// ---------------------------------------------------------------- the CPU reference
// The same integer path: exact integer block sums, then the float rescale, in double, from
// the very Q8_1 blocks the provided quantiser produced (checked against the CPU's own Q8_1
// first). mag is sum over blocks of |d_w| (|sumi d_x| + 8 |s_x|), for the tolerance.
static const double kRtol = 1e-5;

static void cpuIntegerPath(const std::vector<BlockQ4_0>& W, const std::vector<BlockQ8_1>& Xq, int batch, int in, int out,
                           std::vector<double>& y, std::vector<double>& mag) {
  const int nb = in / QK4_0;
  y.assign((size_t)batch * out, 0.0);
  mag.assign((size_t)batch * out, 0.0);
  for (int b = 0; b < batch; ++b) {
    for (int o = 0; o < out; ++o) {
      double acc = 0.0, m = 0.0;
      for (int k = 0; k < nb; ++k) {
        const BlockQ4_0& wb = W[(size_t)o * nb + k];
        const BlockQ8_1& xb = Xq[(size_t)b * nb + k];
        long long sumi = 0;
        for (int j = 0; j < 16; ++j) {
          sumi += (long long)(wb.qs[j] & 0x0F) * xb.qs[j];        // weight j
          sumi += (long long)(wb.qs[j] >> 4) * xb.qs[j + 16];      // weight j + 16
        }
        const double dw = qref::halfToFloat(wb.d), dx = qref::halfToFloat(xb.d), sx = qref::halfToFloat(xb.s);
        acc += dw * ((double)sumi * dx - 8.0 * sx);
        m += std::fabs(dw) * (std::fabs((double)sumi * dx) + 8.0 * std::fabs(sx));
      }
      y[(size_t)b * out + o] = acc;
      mag[(size_t)b * out + o] = m;
    }
  }
}

// The float path, for information: dequantised weights times the float activations.
static void cpuFloatPath(const std::vector<BlockQ4_0>& W, const std::vector<float>& X, int batch, int in, int out,
                         std::vector<double>& y, std::vector<double>& mag) {
  const int nb = in / QK4_0;
  std::vector<float> row(in);
  y.assign((size_t)batch * out, 0.0);
  mag.assign((size_t)batch * out, 0.0);
  for (int o = 0; o < out; ++o) {
    qref::dequantizeQ4_0Row(&W[(size_t)o * nb], row.data(), in);
    for (int b = 0; b < batch; ++b) {
      double acc = 0.0, m = 0.0;
      for (int k = 0; k < in; ++k) {
        const double t = (double)row[k] * X[(size_t)b * in + k];
        acc += t;
        m += std::fabs(t);
      }
      y[(size_t)b * out + o] = acc;
      mag[(size_t)b * out + o] = m;
    }
  }
}

// The provided quantiser against the CPU's Q8_1. A GPU may divide or round one ulp away from
// the CPU (Metal's default math is not IEEE-strict: ch05-facts C7), so a code may be off by
// one and a scale by one FP16 step; anything more is a real disagreement.
static std::string checkQuantiser(const std::vector<BlockQ8_1>& got, const std::vector<float>& X, int batch, int in) {
  std::vector<BlockQ8_1> want((size_t)batch * in / QK8_1);
  for (int b = 0; b < batch; ++b) qref::quantizeQ8_1Row(&X[(size_t)b * in], &want[(size_t)b * in / QK8_1], in);
  for (size_t k = 0; k < want.size(); ++k) {
    const int dd = std::abs((int)got[k].d - (int)want[k].d), ds = std::abs((int)got[k].s - (int)want[k].s);
    bool codes = true;
    for (int j = 0; j < QK8_1; ++j) codes = codes && std::abs((int)got[k].qs[j] - (int)want[k].qs[j]) <= 1;
    if (dd > 1 || ds > 1 || !codes) {
      char b[300];
      std::snprintf(b, sizeof b,
                    "the PROVIDED quantiser (launchQuantizeQ8_1 / quantize_q8_1) disagrees with the CPU's Q8_1 at "
                    "block %zu (d bits %04x vs %04x, s bits %04x vs %04x): it is not one of the defects; did it change?",
                    k, got[k].d, want[k].d, got[k].s, want[k].s);
      return b;
    }
  }
  return "";
}

// ---------------------------------------------------------------- one test case
struct Case {
  std::string name;
  Kind kind;
  int batch, in, out;
};

static std::string runCase(const Case& c, uint32_t seed, std::string* info) {
  std::string why = setupProblem();
  if (!why.empty()) return why;
  const int nb = c.in / QK4_0;
  const std::vector<BlockQ4_0> W = makeWeights(c.kind, c.out, c.in, seed);
  const std::vector<float> X = makeActs(c.kind, c.batch, c.in, seed + 1);
  const size_t wBytes = W.size() * sizeof(BlockQ4_0), xqBytes = (size_t)c.batch * nb * sizeof(BlockQ8_1);
  const size_t yBytes = (size_t)c.batch * c.out * 4;
  Buf* dW = pmpp::alloc(wBytes);
  Buf* dX = pmpp::alloc(X.size() * 4);
  Buf* dXq = outputBuffer(xqBytes, 0xEE, false);
  Buf* dY = outputBuffer(yBytes, 0, true);
  pmpp::upload(dW, W.data(), wBytes);
  pmpp::upload(dX, X.data(), X.size() * 4);
  bool ok = true;
  exec([&](void* pass) {
    ok = enqQuantize(pass, dX, dXq, c.batch, c.in);
    if (ok) enqGemv(pass, dW, dXq, dY, c.batch, c.in, c.out);
  });
  std::string err = pmpp::take_error();
  std::vector<unsigned char> rawXq = fetch(dXq, xqBytes), rawY = fetch(dY, yBytes);
  std::vector<unsigned char> rawW(wBytes);
  pmpp::download(dW, rawW.data(), wBytes);
  for (Buf* b : {dW, dX, dXq, dY}) pmpp::release(b);
  if (!err.empty()) return err;
  if (!ok) return "encodeQuantizeQ8_1 refused to dispatch (SIMD-group width not a multiple of 32)";
  if (std::memcmp(rawW.data(), W.data(), wBytes) != 0) return "the weights were changed";
  if (!guardIntact(rawXq, xqBytes)) return "something wrote past the end of Xq";
  if (!guardIntact(rawY, yBytes)) return "something wrote past the end of Y";

  std::vector<BlockQ8_1> Xq((size_t)c.batch * nb);
  std::memcpy(Xq.data(), rawXq.data(), xqBytes);
  std::string q = checkQuantiser(Xq, X, c.batch, c.in);
  if (!q.empty()) return q;

  std::vector<float> Y((size_t)c.batch * c.out);
  std::memcpy(Y.data(), rawY.data(), yBytes);
  std::vector<double> ref, mag;
  cpuIntegerPath(W, Xq, c.batch, c.in, c.out, ref, mag);
  size_t bad = 0, first = 0;
  for (size_t i = 0; i < Y.size(); ++i) {
    if (!(std::fabs((double)Y[i] - ref[i]) <= kRtol * (1.0 + mag[i]))) {
      if (bad++ == 0) first = i;
    }
  }
  // For information only: how far the integer path is from the float path.
  std::vector<double> fy, fmag;
  cpuFloatPath(W, X, c.batch, c.in, c.out, fy, fmag);
  double worst = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) worst = std::max(worst, std::fabs(ref[i] - fy[i]) / (1.0 + fmag[i]));
  char b[300];
  std::snprintf(b, sizeof b, "integer path vs float path (CPU, dequantised weights x float activations): "
                "max |diff| / (1 + sum |w x|) = %.3g", worst);
  *info = b;
  if (bad) {
    std::snprintf(b, sizeof b, "%zu of %zu outputs wrong; first at batch row %zu, output %zu: got %.7g, expected %.7g "
                  "(tolerance %.3g)", bad, Y.size(), first / c.out, first % c.out, (double)Y[first], ref[first],
                  kRtol * (1.0 + mag[first]));
    return b;
  }
  return "";
}

// The float Q4_0 GEMV the benchmark compares against (lab X2.2's solution or the stand-in),
// checked here so that the comparison is between two correct kernels.
static std::string runFloatCase(int batch, int in, int out, uint32_t seed) {
  std::string why = floatProblem();
  if (!why.empty()) return why;
  const std::vector<BlockQ4_0> W = makeWeights(RANDOM, out, in, seed);
  const std::vector<float> X = makeActs(RANDOM, batch, in, seed + 1);
  const size_t wBytes = W.size() * sizeof(BlockQ4_0), yBytes = (size_t)batch * out * 4;
  Buf* dW = pmpp::alloc(wBytes);
  Buf* dX = pmpp::alloc(X.size() * 4);
  Buf* dY = outputBuffer(yBytes, 0, true);
  pmpp::upload(dW, W.data(), wBytes);
  pmpp::upload(dX, X.data(), X.size() * 4);
  exec([&](void* pass) { enqFloat(pass, dX, dW, dY, batch, in, out); });
  std::string err = pmpp::take_error();
  std::vector<unsigned char> rawY = fetch(dY, yBytes);
  for (Buf* b : {dW, dX, dY}) pmpp::release(b);
  if (!err.empty()) return err;
  if (!guardIntact(rawY, yBytes)) return "something wrote past the end of Y";
  std::vector<float> Y((size_t)batch * out);
  std::memcpy(Y.data(), rawY.data(), yBytes);
  std::vector<double> ref, mag;
  cpuFloatPath(W, X, batch, in, out, ref, mag);
  for (size_t i = 0; i < Y.size(); ++i) {
    if (!(std::fabs((double)Y[i] - ref[i]) <= 1e-5 * (1.0 + mag[i]))) {
      char b[200];
      std::snprintf(b, sizeof b, "output %zu: got %.7g, expected %.7g", i, (double)Y[i], ref[i]);
      return b;
    }
  }
  return "";
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("float Q4_0 GEMV for the comparison: %s\n", PMPP_Q4_0_FROM);
  const std::vector<Case> cases = {
      {"random weights, batch 1, in 32, out 1 (one block)", RANDOM, 1, 32, 1},
      {"random weights, batch 1, in 256, out 3", RANDOM, 1, 256, 3},
      {"random weights, batch 2, in 512, out 5", RANDOM, 2, 512, 5},
      {"random weights, batch 8, in 1024, out 9", RANDOM, 8, 1024, 9},
      {"worst-case block: every weight code 15, every activation code 127 (sum 60,960)", WORST, 2, 256, 3},
      {"weights j and j + 16 at opposite extremes, batch 3, in 512, out 4", SPLIT16, 3, 512, 4},
      {"power-of-two block scales, batch 4, in 768, out 6", POW2, 4, 768, 6},
  };
  uint32_t seed = 3101;
  for (const Case& c : cases) {
    std::string info;
    const uint32_t s = seed++;
    t.check(c.name, [&]() -> std::string { return runCase(c, s, &info); });
    if (!info.empty()) std::printf("        info: %s\n", info.c_str());
  }
  t.check("provided float Q4_0 GEMV (the benchmark's comparison), batch 3, in 512, out 7",
          [&]() -> std::string { return runFloatCase(3, 512, 7, 3201); });
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Weights are random Q4_0 blocks (random codes, scales near 0.01): the kernels' work does not
// depend on the values. Successive timed runs read different copies of the weights, so that a
// run does not find the previous run's weights in a cache; the total size of the copies is
// printed. Apple documents no cache sizes (infra-facts C13), so it is a choice, not a fact.
static const double kWorkingSet = 512e6;

static void randomBlocks(std::vector<BlockQ4_0>& w, uint32_t seed) {
  uint32_t r = seed * 2654435761u + 1u;
  for (BlockQ4_0& b : w) {
    b.d = qref::floatToHalf(0.005f + 0.01f * (float)(xorshift(r) % 1000) / 1000.0f);
    for (int j = 0; j < 16; j += 4) {
      const uint32_t v = xorshift(r);
      std::memcpy(&b.qs[j], &v, 4);
    }
  }
}

static pmpp::Stat measure(const pmpp::Args& a, const std::function<pmpp::Timing()>& once) {
  return pmpp::time_it(std::max(a.reps, 10), once);
}

static int bench(const pmpp::Args& a) {
  std::string why = setupProblem();
  if (!why.empty()) {
    std::printf("cannot benchmark: %s\n", why.c_str());
    return 1;
  }
  const std::string fwhy = floatProblem();
  std::printf("timing: %s; best of at least %d runs after one untimed run\n", timingUnit(), std::max(a.reps, 10));
  std::printf("float path: %s\n", PMPP_Q4_0_FROM);
  const double mem = deviceMemory();
  struct Shape { int in, out; };
  const Shape shapes[] = {{4096, 4096}, {14336, 4096}};
  std::printf("\n  path         batch     in    out   copies   quantise ms     gemv ms    total ms   weights GB/s\n");
  for (const Shape& sh : shapes) {
    const int nb = sh.in / QK4_0;
    const size_t wBytes = (size_t)sh.out * nb * sizeof(BlockQ4_0);
    double budget = kWorkingSet;
    if (mem > 0) budget = std::min(budget, mem / 4);
    const int copies = std::max(1, std::min(64, (int)(budget / (double)wBytes)));
    std::vector<Buf*> W;
    std::vector<BlockQ4_0> host((size_t)sh.out * nb);
    pmpp::take_error();
    for (int c = 0; c < copies; ++c) {
      randomBlocks(host, 4001 + c);
      Buf* b = pmpp::alloc(wBytes);
      pmpp::upload(b, host.data(), wBytes);
      W.push_back(b);
    }
    if (!pmpp::take_error().empty()) {
      std::printf("  in %d, out %d: allocation failed\n", sh.in, sh.out);
      for (Buf* b : W) pmpp::release(b);
      continue;
    }
    for (int batch : {1, 8}) {
      Buf* X = pmpp::alloc((size_t)batch * sh.in * 4);
      Buf* Xq = pmpp::alloc((size_t)batch * nb * sizeof(BlockQ8_1));
      Buf* Y = pmpp::alloc((size_t)batch * sh.out * 4);
      std::vector<float> x = pmpp::random_floats((size_t)batch * sh.in, 77 + batch);
      pmpp::upload(X, x.data(), x.size() * 4);
      int next = 0;
      bool ok = true;
      // (a) the integer path: quantise, GEMV, and both in one pass
      const pmpp::Stat q = measure(a, [&] { return exec([&](void* p) { ok = enqQuantize(p, X, Xq, batch, sh.in) && ok; }); });
      const pmpp::Stat g = measure(a, [&] {
        Buf* w = W[next++ % copies];
        return exec([&](void* p) { enqGemv(p, w, Xq, Y, batch, sh.in, sh.out); });
      });
      const pmpp::Stat both = measure(a, [&] {
        Buf* w = W[next++ % copies];
        return exec([&](void* p) {
          ok = enqQuantize(p, X, Xq, batch, sh.in) && ok;
          enqGemv(p, w, Xq, Y, batch, sh.in, sh.out);
        });
      });
      std::string err = pmpp::take_error();
      if (!err.empty() || !ok) {
        std::printf("  int8_q8_1 batch %d: failed (%s)\n", batch, err.empty() ? "the quantiser refused to dispatch" : err.c_str());
      } else {
        std::printf("  int8_q8_1   %5d  %5d  %5d   %6d   %11.4f  %10.4f  %10.4f   %12.1f\n", batch, sh.in, sh.out, copies,
                    q.min_s * 1e3, g.min_s * 1e3, both.min_s * 1e3, wBytes / g.min_s / 1e9);
        pmpp::bench_row(a, "qhunt").s("path", "int8_q8_1").n("batch", batch).n("in", sh.in).n("out", sh.out)
            .n("seconds_quantise", q.min_s).n("seconds_gemv", g.min_s).n("seconds_total", both.min_s)
            .n("weight_bytes", (double)wBytes).n("copies", copies).s("source", "this lab").emit();
      }
      // (b) the float path: dequantise inside the GEMV, float activations, no quantise pass
      if (fwhy.empty()) {
        const pmpp::Stat f = measure(a, [&] {
          Buf* w = W[next++ % copies];
          return exec([&](void* p) { enqFloat(p, X, w, Y, batch, sh.in, sh.out); });
        });
        err = pmpp::take_error();
        if (!err.empty()) {
          std::printf("  float_q4_0 batch %d: failed (%s)\n", batch, err.c_str());
        } else {
          std::printf("  float_q4_0  %5d  %5d  %5d   %6d   %11s  %10.4f  %10.4f   %12.1f\n", batch, sh.in, sh.out, copies,
                      "-", f.min_s * 1e3, f.min_s * 1e3, wBytes / f.min_s / 1e9);
          pmpp::bench_row(a, "qhunt").s("path", "float_q4_0").n("batch", batch).n("in", sh.in).n("out", sh.out)
              .n("seconds_quantise", 0.0).n("seconds_gemv", f.min_s).n("seconds_total", f.min_s)
              .n("weight_bytes", (double)wBytes).n("copies", copies).s("source", PMPP_Q4_0_FROM).emit();
        }
      } else {
        std::printf("  float_q4_0: not measured (%s)\n", fwhy.c_str());
      }
      for (Buf* b : {X, Xq, Y}) pmpp::release(b);
    }
    std::printf("  (in %d, out %d: %d copies of the weights, %.1f MB in all)\n", sh.in, sh.out, copies,
                copies * (double)wBytes / 1e6);
    for (Buf* b : W) pmpp::release(b);
  }
  std::printf("\nThese are measurements of this machine. 'weights GB/s' is the Q4_0 weight bytes over the GEMV's\n"
              "time, ignoring the activations and the output.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
