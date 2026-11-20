// Lab 22.2 driver (milestone M8). Tests your batched greedy decode step against a CPU
// reference, then measures it at batch sizes from 1 upward. You do not need to edit this
// file; your work goes in work/.
//
//   decode.h                the model, the state, your entry points, the provided pieces
//   provided/pieces.cmake   copies the earlier labs' reference solutions into the build
//   provided/step_timer.cu  one pair of events around a stretch of launches (CUDA, HIP)
//   cpuDecodeStep (below)   the CPU reference, in double
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "decode.h"
#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---------------------------------------------------------------- the GEMMs a step can use
enum Gemm { M2, M3, GEMV };
static const Gemm kGemms[] = {M2, M3, GEMV};
static const char* gemmName(Gemm g) { return g == M2 ? "m2" : g == M3 ? "m3" : "gemv"; }
static const char* gemmSource(Gemm g) {
  return g == M2 ? "lab 6.4's tiled kernel" : g == M3 ? "lab 15.6's register-tiled kernel" : "lab 22.1's gather";
}
// Lab 22.1's kernel is a matrix-vector product: the step uses it for small batches only.
static const int kGemvMaxBatch = 8;
static bool gemmTakes(Gemm g, int batch) { return g != GEMV || batch <= kGemvMaxBatch; }

enum Piece { EMBED, LINEAR, BIAS, RESIDUAL, LM_HEAD, SOFTMAX, ARGMAX, NUM_PIECES };
static const char* kPieceNames[NUM_PIECES] = {"embed", "linear", "bias", "residual", "lm_head", "softmax", "argmax"};

static const float kTemperature = 0.8f;   // softmax scale 1/0.8: a wrong scale cannot pass for a right one

// ---------------------------------------------------------------- device buffers
// Every output buffer is followed by a guard band of kGuard 4-byte words holding kSentinel;
// outputs start as NaN. The weights are allocated to their exact size, so that on the
// emulator a read past one is an AddressSanitizer error.
static const int kGuard = 64;
static const uint32_t kSentinel = 0x5EEDF00Du;
static const uint32_t kQuietNaN = 0x7FC00000u;

struct DevModel {
  int d = 0, vocab = 0, layers = 0;
  Buf* E = nullptr;
  std::vector<Buf*> W, b;
  bool ok = false;
  DevModel(int d_, int vocab_, int layers_) : d(d_), vocab(vocab_), layers(layers_) {
    pmpp::take_error();
    E = pmpp::alloc((size_t)vocab * d * 4);
    for (int l = 0; l < layers; ++l) {
      W.push_back(pmpp::alloc((size_t)d * d * 4));
      b.push_back(pmpp::alloc((size_t)d * 4));
    }
    ok = pmpp::take_error().empty();
  }
  DevModel(const DevModel&) = delete;
  ~DevModel() {
    pmpp::release(E);
    for (Buf* x : W) pmpp::release(x);
    for (Buf* x : b) pmpp::release(x);
  }
};

struct DevState {
  int batch, d, vocab;
  Buf *tokens = nullptr, *H = nullptr, *T = nullptr, *logits = nullptr, *probs = nullptr;
  bool ok = false;
  // `poison`: fill the outputs with NaN (tests). Tokens always start as `first`.
  DevState(int batch_, int d_, int vocab_, const std::vector<int>& first, bool poison)
      : batch(batch_), d(d_), vocab(vocab_) {
    pmpp::take_error();
    tokens = alloc(batch);
    H = alloc((size_t)batch * d);
    T = alloc((size_t)batch * d);
    logits = alloc((size_t)batch * vocab);
    probs = alloc((size_t)batch * vocab);
    ok = pmpp::take_error().empty();
    if (!ok) return;
    std::vector<uint32_t> w(batch + kGuard, kSentinel);
    std::memcpy(w.data(), first.data(), (size_t)batch * 4);
    pmpp::upload(tokens, w.data(), w.size() * 4);
    for (Buf* x : {H, T, logits, probs}) fill(x, x == H || x == T ? (size_t)batch * d : (size_t)batch * vocab, poison);
  }
  DevState(const DevState&) = delete;
  ~DevState() {
    for (Buf* x : {tokens, H, T, logits, probs}) pmpp::release(x);
  }
 private:
  static Buf* alloc(size_t n) { return pmpp::alloc((n + kGuard) * 4); }
  // NaN for the tests; zeros for the benchmark, whose step writes every output before it reads it.
  static void fill(Buf* x, size_t n, bool poison) {
    std::vector<uint32_t> w(n + kGuard, poison ? kQuietNaN : 0u);
    for (size_t i = n; i < w.size(); ++i) w[i] = kSentinel;
    pmpp::upload(x, w.data(), w.size() * 4);
  }
};

// n data words plus the guard band behind them.
template <class T>
static std::vector<T> fetch(Buf* b, size_t n) {
  static_assert(sizeof(T) == 4, "4-byte elements");
  std::vector<T> v(n + kGuard);
  pmpp::download(b, v.data(), v.size() * 4);
  return v;
}
template <class T>
static bool guardIntact(const std::vector<T>& v, size_t n) {
  for (size_t i = n; i < v.size(); ++i) {
    uint32_t w;
    std::memcpy(&w, &v[i], 4);
    if (w != kSentinel) return false;
  }
  return true;
}

// ---------------------------------------------------------------- backends
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
using pmpp::metal::buffer;

// The provided pieces' encoders that decode.h does not declare (renamed by pieces.cmake).
void encodeLinearM2(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                    id<MTLBuffer> Y, int32_t batch, int32_t in, int32_t out, int32_t tileWidth);   // lab 6.4's
#if defined(PMPP_M3_METAL)
void encodeLinearM3(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                    id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);                    // lab 15.6's
#endif
#if defined(PMPP_GEMV_METAL)
void encodeGemvGather(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> X, id<MTLBuffer> W,
                      id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out);                  // lab 22.1's
#endif

static bool gemmPresent(Gemm g, std::string* why) {
  if (g == M2) return true;
#if defined(PMPP_M3_METAL)
  if (g == M3) return true;
#endif
#if defined(PMPP_GEMV_METAL)
  if (g == GEMV) return true;
#endif
  *why = g == M3 ? "lab 15.6's solution was not found when the lab was configured"
                 : "lab 22.1's solution, with the signatures provided/pieces.cmake names, was not found when the lab was configured";
  return false;
}

// Lab 6.4's driver compiles its kernel with TILE_WIDTH 16; its encoder takes signed sizes and
// the tile width. This adapter gives it EncodeLinearFn's shape.
static const int kM2Tile = 16;
static void encodeM2(id<MTLComputeCommandEncoder> enc, id<MTLComputePipelineState> p, id<MTLBuffer> X,
                     id<MTLBuffer> W, id<MTLBuffer> Y, uint32_t batch, uint32_t in, uint32_t out) {
  encodeLinearM2(enc, p, X, W, Y, (int32_t)batch, (int32_t)in, (int32_t)out, kM2Tile);
}

// Why pipeline p cannot run `threads` threads per threadgroup here, or "".
static std::string threadsProblem(id<MTLComputePipelineState> p, NSUInteger threads, const std::string& what) {
  char b[300];
  if (!p) return what + " did not build: " + pmpp::take_error();
  if (p.maxTotalThreadsPerThreadgroup < threads) {
    std::snprintf(b, sizeof b, "%s: this pipeline allows %lu threads per threadgroup, the kernel needs %lu", what.c_str(),
                  (unsigned long)p.maxTotalThreadsPerThreadgroup, (unsigned long)threads);
    return b;
  }
  if (p.staticThreadgroupMemoryLength > pmpp::metal::device().maxThreadgroupMemoryLength) {
    std::snprintf(b, sizeof b, "%s: needs %lu bytes of threadgroup memory, this device has %lu", what.c_str(),
                  (unsigned long)p.staticThreadgroupMemoryLength,
                  (unsigned long)pmpp::metal::device().maxThreadgroupMemoryLength);
    return b;
  }
  return "";
}
// Lab 10.3's rule: 256 threads per threadgroup, in whole SIMD-groups, no more groups than lanes.
static std::string m4Problem(id<MTLComputePipelineState> p, const std::string& what) {
  std::string e = threadsProblem(p, 256, what);
  if (!e.empty()) return e;
  const NSUInteger w = p.threadExecutionWidth;
  if (w == 0 || 256 % w != 0 || 256 / w > w) {
    return what + ": SIMD-group width " + std::to_string((unsigned long)w) + " does not divide 256 threads into 1 to " +
           std::to_string((unsigned long)w) + " whole SIMD-groups";
  }
  return "";
}

struct Pieces {
  DecodePipelines pipe;
  EncodeLinearFn encode = nullptr;
  std::string problem;
};
// The pipelines a step with GEMM g needs, each from its own lab's kernel.metal, compiled with
// the constants that lab's driver uses (pmpp::metal::pipeline caches them).
static Pieces piecesFor(Gemm g) {
  Pieces P;
  DecodePipelines& p = P.pipe;
  auto first = [&](const std::string& e) { if (P.problem.empty() && !e.empty()) P.problem = e; };
  p.embed = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "embed_rows");
  first(p.embed ? std::string() : "your kernel.metal did not build: " + pmpp::take_error());
  p.addBias = pmpp::metal::pipeline(PMPP_M0_METAL, "add_bias");
  first(p.addBias ? std::string() : "lab 2.3's add_bias did not build: " + pmpp::take_error());
  p.addResidual = pmpp::metal::pipeline(PMPP_M0_METAL, "add_residual");
  first(p.addResidual ? std::string() : "lab 2.3's add_residual did not build: " + pmpp::take_error());
  p.softmax = pmpp::metal::pipeline(PMPP_M4_METAL, "softmax_rows");
  first(m4Problem(p.softmax, "lab 10.3's softmax_rows"));
  p.argmax = pmpp::metal::pipeline(PMPP_M4_METAL, "argmax_rows");
  first(m4Problem(p.argmax, "lab 10.3's argmax_rows"));
  switch (g) {
    case M2:
      p.linear = pmpp::metal::pipeline(PMPP_M2_METAL, "linear", {{"TILE_WIDTH", std::to_string(kM2Tile)}});
      first(threadsProblem(p.linear, (NSUInteger)kM2Tile * kM2Tile, "lab 6.4's linear"));
      P.encode = encodeM2;
      break;
    case M3: {
#if defined(PMPP_M3_METAL)
      // As lab 15.6's driver: the SIMD-group width is the pipeline's threadExecutionWidth
      // (Apple documents no fixed width, ch15-facts C1); 64 if it reports 64, else 32.
      id<MTLComputePipelineState> probe = pmpp::metal::pipeline(PMPP_M3_METAL, "linear_m3", {{"SIMD_WIDTH", "32"}});
      const int width = (probe && probe.threadExecutionWidth == 64) ? 64 : 32;
      p.linear = pmpp::metal::pipeline(PMPP_M3_METAL, "linear_m3", {{"SIMD_WIDTH", std::to_string(width)}});
      first(threadsProblem(p.linear, 256, "lab 15.6's linear_m3"));
      P.encode = encodeLinearM3;
#endif
      break;
    }
    case GEMV:
#if defined(PMPP_GEMV_METAL)
      p.linear = pmpp::metal::pipeline(PMPP_GEMV_METAL, PMPP_GEMV_KERNEL);
      first(p.linear ? std::string() : "lab 22.1's " PMPP_GEMV_KERNEL " did not build: " + pmpp::take_error());
      P.encode = encodeGemvGather;
#endif
      break;
  }
  if (!P.encode) first(std::string("no ") + gemmName(g) + " on this build");
  return P;
}

static id<MTLComputeCommandEncoder> encoderOf(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }

// A model and a GEMM, ready to run steps and pieces. Each call is one command buffer.
struct Runner {
  DecodeModel m{};
  std::string problem;
  Runner(DevModel& dm, Gemm g) {
    Pieces P = piecesFor(g);
    problem = P.problem;
    m.d = (uint32_t)dm.d;
    m.vocab = (uint32_t)dm.vocab;
    m.layers = (uint32_t)dm.layers;
    m.E = buffer(dm.E);
    NSMutableArray<id<MTLBuffer>>* W = [[NSMutableArray alloc] initWithCapacity:dm.layers];
    NSMutableArray<id<MTLBuffer>>* B = [[NSMutableArray alloc] initWithCapacity:dm.layers];
    for (int l = 0; l < dm.layers; ++l) {
      [W addObject:buffer(dm.W[l])];
      [B addObject:buffer(dm.b[l])];
    }
    m.W = W;
    m.b = B;
    m.encodeLinear = P.encode;
    m.pipelines = P.pipe;
  }
  Runner(const Runner&) = delete;
  static DecodeState view(DevState& s) {
    DecodeState v{};
    v.batch = (uint32_t)s.batch;
    v.temperature = kTemperature;
    v.tokens = buffer(s.tokens);
    v.H = buffer(s.H);
    v.T = buffer(s.T);
    v.logits = buffer(s.logits);
    v.probs = buffer(s.probs);
    return v;
  }
  // One step for every sequence of s.
  pmpp::Timing step(DevState& s, bool* ok) {
    DecodeState v = view(s);
    *ok = true;
    return pmpp::run([&](void* pass) { *ok = encodeDecodeStep(encoderOf(pass), m, v); });
  }
  // One step for each of `count` states, one after another, in one command buffer.
  pmpp::Timing steps(std::vector<std::unique_ptr<DevState>>& states, int count, bool* ok) {
    std::vector<DecodeState> v;
    for (int i = 0; i < count; ++i) v.push_back(view(*states[i]));
    *ok = true;
    return pmpp::run([&](void* pass) {
      for (int i = 0; i < count; ++i) *ok = encodeDecodeStep(encoderOf(pass), m, v[i]) && *ok;
    });
  }
  pmpp::Timing embed(DevState& s) {
    return pmpp::run([&](void* pass) {
      encodeEmbed(encoderOf(pass), m.pipelines.embed, m.E, buffer(s.tokens), buffer(s.H), (uint32_t)s.batch, m.d);
    });
  }
  // One piece of the step, alone (the benchmark's 'piece' rows).
  pmpp::Timing piece(Piece which, DevState& s) {
    const DecodePipelines& p = m.pipelines;
    const uint32_t B = (uint32_t)s.batch;
    DecodeState v = view(s);
    return pmpp::run([&](void* pass) {
      id<MTLComputeCommandEncoder> enc = encoderOf(pass);
      switch (which) {
        case EMBED: encodeEmbed(enc, p.embed, m.E, v.tokens, v.H, B, m.d); break;
        case LINEAR: m.encodeLinear(enc, p.linear, v.H, m.W[0], v.T, B, m.d, m.d); break;
        case BIAS: encodeAddBias(enc, p.addBias, v.T, v.T, m.b[0], B, m.d); break;
        case RESIDUAL: encodeAddResidual(enc, p.addResidual, v.H, v.T, B * m.d); break;
        case LM_HEAD: m.encodeLinear(enc, p.linear, v.H, m.E, v.logits, B, m.d, m.vocab); break;
        case SOFTMAX: encodeSoftmaxRows(enc, p.softmax, v.logits, v.probs, B, m.vocab, 1.0f / kTemperature); break;
        default: encodeArgmaxRows(enc, p.argmax, v.logits, v.tokens, B, m.vocab); break;
      }
    });
  }
};

static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* stepUnit() { return "one command buffer per step, GPUEndTime - GPUStartTime"; }

#else   // CUDA, HIP, emulators
// The provided pieces decode.h does not declare (renamed by pieces.cmake).
void launchLinearM2(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 6.4's
#if defined(PMPP_HAVE_M3)
void launchLinearM3(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 15.6's
#endif
#if defined(PMPP_HAVE_GEMV)
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out); // lab 22.1's
#endif
float eventTimeMs(void (*enqueue)(void* ctx), void* ctx);                                     // provided/step_timer.cu

static LinearFn linearFor(Gemm g) {
  switch (g) {
    case M2: return launchLinearM2;
#if defined(PMPP_HAVE_M3)
    case M3: return launchLinearM3;
#endif
#if defined(PMPP_HAVE_GEMV)
    case GEMV: return launchGemvGather;
#endif
    default: return nullptr;
  }
}
static bool gemmPresent(Gemm g, std::string* why) {
  if (linearFor(g)) return true;
  *why = g == M3 ? "lab 15.6's solution was not found when the lab was configured"
                 : "lab 22.1's solution, with the signature provided/pieces.cmake names, was not found when the lab was configured";
  return false;
}

static float* f(Buf* b) { return (float*)pmpp::raw(b); }
static int* i32(Buf* b) { return (int*)pmpp::raw(b); }

// Runs `work` between two events (provided/step_timer.cu) inside pmpp::run, which checks
// for launch errors and waits. The device's event time replaces the host's clock.
static pmpp::Timing exec(const std::function<void()>& work) {
  float ms = -1.0f;
  pmpp::Timing t = pmpp::run([&](void*) {
    ms = eventTimeMs([](void* c) { (*(const std::function<void()>*)c)(); }, (void*)&work);
  });
  if (ms >= 0) t.gpu_s = ms * 1e-3;
  return t;
}

struct Runner {
  DecodeModel m{};
  std::vector<const float*> W, b;   // host arrays of device pointers, as DecodeModel wants them
  std::string problem;
  Runner(DevModel& dm, Gemm g) {
    for (int l = 0; l < dm.layers; ++l) {
      W.push_back(f(dm.W[l]));
      b.push_back(f(dm.b[l]));
    }
    m.d = dm.d;
    m.vocab = dm.vocab;
    m.layers = dm.layers;
    m.E = f(dm.E);
    m.W = W.data();
    m.b = b.data();
    m.linear = linearFor(g);
    if (!m.linear) problem = std::string("no ") + gemmName(g) + " on this build";
  }
  Runner(const Runner&) = delete;
  static DecodeState view(DevState& s) {
    DecodeState v{};
    v.batch = s.batch;
    v.temperature = kTemperature;
    v.tokens = i32(s.tokens);
    v.H = f(s.H);
    v.T = f(s.T);
    v.logits = f(s.logits);
    v.probs = f(s.probs);
    return v;
  }
  pmpp::Timing step(DevState& s, bool* ok) {
    *ok = true;
    DecodeState v = view(s);
    return exec([&] { launchDecodeStep(m, v); });
  }
  pmpp::Timing steps(std::vector<std::unique_ptr<DevState>>& states, int count, bool* ok) {
    *ok = true;
    std::vector<DecodeState> v;
    for (int i = 0; i < count; ++i) v.push_back(view(*states[i]));
    return exec([&] {
      for (int i = 0; i < count; ++i) launchDecodeStep(m, v[i]);
    });
  }
  pmpp::Timing embed(DevState& s) {
    return exec([&] { launchEmbed(m.E, i32(s.tokens), f(s.H), s.batch, m.d); });
  }
  pmpp::Timing piece(Piece which, DevState& s) {
    DecodeState v = view(s);
    const int B = s.batch;
    return exec([&] {
      switch (which) {
        case EMBED: launchEmbed(m.E, v.tokens, v.H, B, m.d); break;
        case LINEAR: m.linear(v.H, m.W[0], v.T, B, m.d, m.d); break;
        case BIAS: launchAddBias(v.T, v.T, m.b[0], B, m.d); break;
        case RESIDUAL: launchAddResidual(v.H, v.T, B * m.d); break;
        case LM_HEAD: m.linear(v.H, m.E, v.logits, B, m.d, m.vocab); break;
        case SOFTMAX: launchSoftmaxRows(v.logits, v.probs, B, m.vocab, 1.0f / kTemperature); break;
        default: launchArgmaxRows(v.logits, v.tokens, B, m.vocab); break;
      }
    });
  }
};

static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* stepUnit() { return "one pair of events per step"; }
#endif

// ---------------------------------------------------------------- the model, on the host
struct HostModel {
  int d = 0, vocab = 0, layers = 0;
  std::vector<float> E;                      // [vocab x d]
  std::vector<std::vector<float>> W, b;      // W[l]: [d x d] (out, in); b[l]: [d]
};

// E uniform in [-1, 1]; W scaled so that a layer changes H by a fraction of itself; small biases.
static const float kBiasScale = 0.1f;
static float weightScale(int d) { return 0.5f / std::sqrt((float)d); }

static HostModel makeModel(int d, int vocab, int layers, uint32_t seed) {
  HostModel m;
  m.d = d;
  m.vocab = vocab;
  m.layers = layers;
  m.E = pmpp::random_floats((size_t)vocab * d, seed);
  for (int l = 0; l < layers; ++l) {
    std::vector<float> w = pmpp::random_floats((size_t)d * d, seed + 101 + l);
    for (float& x : w) x *= weightScale(d);
    std::vector<float> bb = pmpp::random_floats((size_t)d, seed + 201 + l);
    for (float& x : bb) x *= kBiasScale;
    m.W.push_back(std::move(w));
    m.b.push_back(std::move(bb));
  }
  return m;
}

static void uploadModel(DevModel& dm, const HostModel& h) {
  pmpp::upload(dm.E, h.E.data(), h.E.size() * 4);
  for (int l = 0; l < h.layers; ++l) {
    pmpp::upload(dm.W[l], h.W[l].data(), h.W[l].size() * 4);
    pmpp::upload(dm.b[l], h.b[l].data(), h.b[l].size() * 4);
  }
}

// ---------------------------------------------------------------- the CPU reference
// Tolerances. H and the logits: |got - ref| <= kRtol * (1 + mag), where mag is the sum of
// the absolute values of every product and bias that went into the element (the harness's
// rule). The probabilities are checked against the softmax of the logits your step computed,
// within lab 10.3's tolerance for its softmax.
static const double kRtol = 1e-4;
static const double kProbRtol = 1e-4, kProbAtol = 1e-20;
// The test model is chosen so that, at every step it is tested on, the reference's top logit
// beats the runner-up by more than kGapRtol * (2 + mag of the two): far more than float32
// rounding moves a logit, so a correct step's argmax cannot pick the runner-up.
static const double kGapRtol = 1e-5;

struct RefStep {
  int next = -1;                        // argmax of the logits, the lowest index on a tie (M4's rule)
  double gap = 0, gapNeeded = 0;        // top logit minus the runner-up, and what the tests require
  std::vector<double> H, magH;          // H after the layers, and its magnitudes
  std::vector<double> logits, magL;
};

// One greedy decode step for one sequence whose current token is `token`, in double, from
// the same float weights. With no attention the step depends on the current token alone,
// so the tests compute it once per token (refFor).
static void cpuDecodeStep(const HostModel& m, int token, RefStep& out) {
  const int d = m.d;
  out.H.assign(d, 0.0);
  out.magH.assign(d, 0.0);
  for (int j = 0; j < d; ++j) {
    out.H[j] = m.E[(size_t)token * d + j];
    out.magH[j] = std::fabs(out.H[j]);
  }
  std::vector<double> T(d), magT(d);
  for (int l = 0; l < m.layers; ++l) {
    const float* W = m.W[l].data();
    for (int o = 0; o < d; ++o) {                          // T = H W^T + b
      double acc = 0.0, mag = 0.0;
      for (int k = 0; k < d; ++k) {
        const double w = W[(size_t)o * d + k];
        acc += out.H[k] * w;
        mag += out.magH[k] * std::fabs(w);
      }
      T[o] = acc + m.b[l][o];
      magT[o] = mag + std::fabs((double)m.b[l][o]);
    }
    for (int j = 0; j < d; ++j) {                          // H += T
      out.H[j] += T[j];
      out.magH[j] += magT[j];
    }
  }
  out.logits.assign(m.vocab, 0.0);
  out.magL.assign(m.vocab, 0.0);
  for (int v = 0; v < m.vocab; ++v) {                      // logits = H E^T
    const float* e = &m.E[(size_t)v * d];
    double acc = 0.0, mag = 0.0;
    for (int j = 0; j < d; ++j) {
      acc += out.H[j] * e[j];
      mag += out.magH[j] * std::fabs((double)e[j]);
    }
    out.logits[v] = acc;
    out.magL[v] = mag;
  }
  int best = 0;
  for (int v = 1; v < m.vocab; ++v)
    if (out.logits[v] > out.logits[best]) best = v;        // strictly greater: ties keep the lower index
  int second = best == 0 ? 1 : 0;
  for (int v = 0; v < m.vocab; ++v)
    if (v != best && out.logits[v] > out.logits[second]) second = v;
  out.next = best;
  out.gap = m.vocab > 1 ? out.logits[best] - out.logits[second] : 1e300;
  out.gapNeeded = m.vocab > 1 ? kGapRtol * (2.0 + out.magL[best] + out.magL[second]) : 0.0;
}

struct TestModel {
  HostModel host;
  std::map<int, RefStep> ref;   // the reference step by current token, computed on demand
  uint32_t seed = 0;
  int reseeds = 0;
};
static const RefStep& refFor(TestModel& tm, int token) {
  auto it = tm.ref.find(token);
  if (it != tm.ref.end()) return it->second;
  RefStep& r = tm.ref[token];
  cpuDecodeStep(tm.host, token, r);
  return r;
}

// The first tokens of a batch: 0 and vocab - 1 (the ends of the embedding), then spread out.
static std::vector<int> startTokens(int batch, int vocab) {
  std::vector<int> t(batch);
  for (int r = 0; r < batch; ++r) t[r] = r == 0 ? 0 : r == 1 ? vocab - 1 : (int)(((long long)r * 7919 + 11) % vocab);
  return t;
}

// A model whose every tested step has a clear winner (kGapRtol): re-seeded until it does.
static std::unique_ptr<TestModel> pickTestModel(int d, int vocab, int layers, const std::vector<int>& batches,
                                                int steps) {
  const uint32_t base = 2201;
  for (uint32_t seed = base;; ++seed) {
    std::unique_ptr<TestModel> tm(new TestModel());
    tm->host = makeModel(d, vocab, layers, seed);
    tm->seed = seed;
    tm->reseeds = (int)(seed - base);
    bool clear = true;
    for (int B : batches) {
      for (int tok : startTokens(B, vocab)) {
        for (int s = 0; s < steps && clear; ++s) {
          const RefStep& r = refFor(*tm, tok);
          clear = r.gap > r.gapNeeded;
          tok = r.next;
        }
      }
      if (!clear) break;
    }
    if (clear || seed - base >= 100) return tm;   // after 100 seeds, test anyway (and say so)
  }
}

// ---------------------------------------------------------------- tests
static std::string at(int step) { return "after step " + std::to_string(step) + ": "; }

static std::string weightsUnchanged(DevModel& dm, const HostModel& h) {
  auto same = [](Buf* b, const std::vector<float>& want) {
    std::vector<float> got(want.size());
    pmpp::download(b, got.data(), got.size() * 4);
    return std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
  };
  if (!same(dm.E, h.E)) return "the embedding E was changed (it is an input, and the LM head's weight)";
  for (int l = 0; l < h.layers; ++l) {
    if (!same(dm.W[l], h.W[l])) return "W[" + std::to_string(l) + "] was changed";
    if (!same(dm.b[l], h.b[l])) return "b[" + std::to_string(l) + "] was changed";
  }
  return "";
}

static std::string guardsIntact(DevState& s) {
  const size_t bd = (size_t)s.batch * s.d, bv = (size_t)s.batch * s.vocab;
  if (!guardIntact(fetch<int>(s.tokens, s.batch), s.batch)) return "something wrote past the end of tokens";
  if (!guardIntact(fetch<float>(s.H, bd), bd)) return "something wrote past the end of H";
  if (!guardIntact(fetch<float>(s.T, bd), bd)) return "something wrote past the end of T";
  if (!guardIntact(fetch<float>(s.logits, bv), bv)) return "something wrote past the end of logits";
  if (!guardIntact(fetch<float>(s.probs, bv), bv)) return "something wrote past the end of probs";
  return "";
}

static std::string describeElement(const char* what, int r, int c, const char* colName, double got, double want,
                                   size_t bad) {
  char b[300];
  std::snprintf(b, sizeof b, "%s wrong in %zu element%s; first at sequence %d, %s %d: got %.7g, expected %.7g", what, bad,
                bad == 1 ? "" : "s", r, colName, c, got, want);
  return b;
}

// After one step: H, logits and probs of every sequence against the reference for the token
// it had before the step, and the tokens it wrote.
static std::string checkStep(TestModel& tm, DevState& s, const std::vector<int>& before, int step) {
  const HostModel& h = tm.host;
  const int B = s.batch, d = h.d, V = h.vocab;
  std::vector<int> toks = fetch<int>(s.tokens, B);
  std::vector<float> H = fetch<float>(s.H, (size_t)B * d);
  std::vector<float> L = fetch<float>(s.logits, (size_t)B * V);
  std::vector<float> P = fetch<float>(s.probs, (size_t)B * V);
  std::string g = guardsIntact(s);
  if (!g.empty()) return at(step) + g;

  struct Bad { size_t n = 0; int r = 0, c = 0; double got = 0, want = 0; };
  Bad bh, bl, bp;
  auto note = [](Bad& b, int r, int c, double got, double want) {
    if (b.n++ == 0) { b.r = r; b.c = c; b.got = got; b.want = want; }
  };
  const double scale = (double)(1.0f / kTemperature);   // the scale the step passes, in float
  for (int r = 0; r < B; ++r) {
    const RefStep& ref = refFor(tm, before[r]);
    for (int j = 0; j < d; ++j) {
      const double got = H[(size_t)r * d + j];
      if (!(std::fabs(got - ref.H[j]) <= kRtol * (1.0 + ref.magH[j]))) note(bh, r, j, got, ref.H[j]);
    }
    for (int v = 0; v < V; ++v) {
      const double got = L[(size_t)r * V + v];
      if (!(std::fabs(got - ref.logits[v]) <= kRtol * (1.0 + ref.magL[v]))) note(bl, r, v, got, ref.logits[v]);
    }
    // softmax(scale * the step's own logits), in double
    double mx = -1e300;
    for (int v = 0; v < V; ++v) mx = std::max(mx, scale * L[(size_t)r * V + v]);
    double z = 0.0;
    for (int v = 0; v < V; ++v) z += std::exp(scale * L[(size_t)r * V + v] - mx);
    for (int v = 0; v < V; ++v) {
      const double want = std::exp(scale * L[(size_t)r * V + v] - mx) / z;
      const double got = P[(size_t)r * V + v];
      if (!(std::fabs(got - want) <= kProbAtol + kProbRtol * want)) note(bp, r, v, got, want);
    }
  }
  if (bh.n) return at(step) + describeElement("H", bh.r, bh.c, "feature", bh.got, bh.want, bh.n);
  if (bl.n) return at(step) + describeElement("logits", bl.r, bl.c, "token", bl.got, bl.want, bl.n);
  if (bp.n) {
    return at(step) + describeElement("probs", bp.r, bp.c, "token", bp.got, bp.want, bp.n) +
           " (expected: the softmax of your logits times 1/temperature)";
  }
  for (int r = 0; r < B; ++r) {
    const RefStep& ref = refFor(tm, before[r]);
    if (toks[r] != ref.next) {
      char b[300];
      std::snprintf(b, sizeof b, "sequence %d was on token %d; it got next token %d, the reference's argmax is %d",
                    r, before[r], toks[r], ref.next);
      return at(step) + b;
    }
  }
  return "";
}

// `steps` greedy steps of a batch, every one checked, then the weights. `tokens` receives
// the tokens after each step that passed (the batch-invariance check reuses them).
static std::string checkDecode(Gemm g, TestModel& tm, DevModel& dm, int batch, int steps,
                               std::vector<std::vector<int>>* tokens) {
  tokens->clear();
  Runner run(dm, g);
  if (!run.problem.empty()) return run.problem;
  const HostModel& h = tm.host;
  std::vector<int> tok = startTokens(batch, h.vocab);
  DevState s(batch, h.d, h.vocab, tok, true);
  if (!s.ok) return "could not allocate the state";
  for (int step = 1; step <= steps; ++step) {
    bool ok = true;
    run.step(s, &ok);
    std::string err = pmpp::take_error();
    if (!err.empty()) return at(step) + err;
    if (!ok) return at(step) + "encodeDecodeStep returned false: a piece refused to dispatch";
    std::string e = checkStep(tm, s, tok, step);
    if (!e.empty()) return e;
    for (int r = 0; r < batch; ++r) tok[r] = refFor(tm, tok[r]).next;
    tokens->push_back(tok);
  }
  return weightsUnchanged(dm, h);
}

// The tokens of `first.size()` sequences decoded together, step by step.
static std::string decodeTokens(Runner& run, const HostModel& h, const std::vector<int>& first, int steps,
                                std::vector<std::vector<int>>& out) {
  DevState s((int)first.size(), h.d, h.vocab, first, true);
  if (!s.ok) return "could not allocate the state";
  for (int step = 1; step <= steps; ++step) {
    bool ok = true;
    run.step(s, &ok);
    std::string err = pmpp::take_error();
    if (!err.empty()) return at(step) + err;
    if (!ok) return at(step) + "encodeDecodeStep returned false: a piece refused to dispatch";
    std::vector<int> t = fetch<int>(s.tokens, first.size());
    t.resize(first.size());
    out.push_back(t);
  }
  return "";
}

// Batch invariance: every sequence of a batch decodes exactly as it does alone. `together`
// is the batch's tokens from its decode check, which must have passed: a step that is wrong
// for the batch says nothing about invariance (one that writes nothing would be "invariant").
static std::string checkInvariance(Gemm g, TestModel& tm, DevModel& dm, int batch, int steps,
                                   const std::vector<std::vector<int>>& together) {
  if ((int)together.size() != steps) return "not checked: this batch's decode check above did not pass";
  Runner run(dm, g);
  if (!run.problem.empty()) return run.problem;
  const std::vector<int> first = startTokens(batch, tm.host.vocab);
  for (int r = 0; r < batch; ++r) {
    std::vector<std::vector<int>> alone;
    std::string e = decodeTokens(run, tm.host, {first[r]}, steps, alone);
    if (!e.empty()) return "sequence " + std::to_string(r) + " alone, " + e;
    for (int step = 0; step < steps; ++step) {
      if (alone[step][0] != together[step][r]) {
        char b[300];
        std::snprintf(b, sizeof b, "sequence %d: after step %d it has token %d in the batch of %d, token %d alone", r,
                      step + 1, together[step][r], batch, alone[step][0]);
        return b;
      }
    }
  }
  return "";
}

// launchEmbed / encodeEmbed on its own: rows copied exactly, tokens untouched, nothing past H.
static std::string checkEmbed(TestModel& tm, DevModel& dm, const std::vector<int>& tokens) {
  const HostModel& h = tm.host;
  const int B = (int)tokens.size();
  Runner run(dm, M2);   // only the embedding pipeline is used
  DevState s(B, h.d, h.vocab, tokens, true);
  if (!s.ok) return "could not allocate the state";
#if defined(PMPP_BACKEND_METAL)
  if (!run.m.pipelines.embed) return run.problem;
#endif
  run.embed(s);
  std::string err = pmpp::take_error();
  if (!err.empty()) return err;
  std::vector<float> H = fetch<float>(s.H, (size_t)B * h.d);
  std::vector<int> t = fetch<int>(s.tokens, B);
  for (int r = 0; r < B; ++r) {
    for (int j = 0; j < h.d; ++j) {
      const float want = h.E[(size_t)tokens[r] * h.d + j];
      const float got = H[(size_t)r * h.d + j];
      if (std::memcmp(&got, &want, 4) != 0) {
        char b[300];
        std::snprintf(b, sizeof b, "H[%d][%d] is %.7g; row %d of E (token %d) has %.7g there", r, j, got, tokens[r],
                      tokens[r], want);
        return b;
      }
    }
    if (t[r] != tokens[r]) return "the embedding changed tokens[" + std::to_string(r) + "]";
  }
  std::string g = guardsIntact(s);
  if (!g.empty()) return g;
  return weightsUnchanged(dm, h);
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const bool emu = pmpp::is_emu();
  const int d = emu ? 48 : 256, vocab = emu ? 301 : 4099, layers = 2, steps = emu ? 4 : 6;
  const std::vector<int> batches = emu ? std::vector<int>{1, 2, 3, 5} : std::vector<int>{1, 2, 3, 8, 17, 64};

  std::unique_ptr<TestModel> tm = pickTestModel(d, vocab, layers, batches, steps);
  std::printf("test model: d = %d, vocab = %d, layers = %d, seed %u (%d re-seed%s to give every tested step a clear "
              "argmax); temperature %.2f\n",
              d, vocab, layers, tm->seed, tm->reseeds, tm->reseeds == 1 ? "" : "s", (double)kTemperature);
  if (tm->reseeds >= 100) std::printf("  warning: no seed in 100 gave every step a clear argmax; token checks may be flaky\n");
  DevModel dm(d, vocab, layers);
  if (!dm.ok) {
    t.check("allocate the test model", [] { return std::string("allocation failed"); });
    return t.finish();
  }
  uploadModel(dm, tm->host);

  const std::vector<int> embedA = {0, vocab - 1, 13, 13, 5, 1, vocab - 1};
  t.check("embed alone: batch 7, tokens 0, V-1, 13, 13, 5, 1, V-1 (rows copied exactly)",
          [&] { return checkEmbed(*tm, dm, embedA); });
  t.check("embed alone: batch 1, token V-1", [&] { return checkEmbed(*tm, dm, {vocab - 1}); });

  // On the emulators the batch-invariance check runs once, with the GEMM that costs the
  // emulator least (M3: its 128-wide tiles launch the fewest threads), since invariance is a
  // property of how your step treats the batch, not of the GEMM it calls. On a GPU it runs
  // once per GEMM.
  std::string unused;
  const Gemm invGemm = gemmPresent(M3, &unused) ? M3 : M2;
  for (Gemm g : kGemms) {
    std::string why;
    if (!gemmPresent(g, &why)) {
      std::printf("  (no %s cases: %s)\n", gemmName(g), why.c_str());
      continue;
    }
    std::map<int, std::vector<std::vector<int>>> together;   // batch -> its tokens after each step
    int largest = 0;
    for (int B : batches) {
      if (!gemmTakes(g, B)) continue;
      largest = B;
      t.check(std::string(gemmName(g)) + ", batch " + std::to_string(B) + ": " + std::to_string(steps) +
                  " steps against the CPU reference",
              [&] { return checkDecode(g, *tm, dm, B, steps, &together[B]); });
    }
    // Every sequence of one batch decoded alone again. The largest batch on a GPU; batch 3
    // on the emulators, where each decode of the batch costs seconds.
    const int inv = emu ? 3 : largest;
    if (inv > 1 && together.count(inv) && (!emu || g == invGemm)) {
      t.check(std::string(gemmName(g)) + ", batch " + std::to_string(inv) + ": each sequence decodes as it does alone",
              [&] { return checkInvariance(g, *tm, dm, inv, steps, together[inv]); });
    }
  }
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
// Bytes and FLOPs of one step in the ideal count (ours, notes/today/ch22-facts.md E1, E2):
// every array read or written once; the GEMMs' 2 * B * K * N FLOPs, plus one FLOP per bias
// and residual element. Softmax, argmax and the embedding are counted in bytes only.
static double modelFlops(double B, double d, double V, double layers) {
  return 2.0 * B * d * (layers * d + V) + 2.0 * layers * B * d;
}
static double modelBytes(double B, double d, double V, double layers) {
  const double embed = 2.0 * B * d + B;                                  // E rows in, H out, token ids
  const double layer = (d * d + 2.0 * B * d) + (2.0 * B * d + d) + 3.0 * B * d;   // linear, bias, residual
  const double head = V * d + B * d + B * V;                             // E, H in; logits out
  const double sampler = 2.0 * B * V + B * V + B;                        // softmax in and out; argmax in, ids out
  return 4.0 * (embed + layers * layer + head + sampler);
}

static void uploadRandom(Buf* b, size_t n, uint32_t seed, float scale) {
  std::vector<float> v = pmpp::random_floats(n, seed);
  if (scale != 1.0f)
    for (float& x : v) x *= scale;
  pmpp::upload(b, v.data(), n * 4);
}

static std::vector<int> benchTokens(int batch, int vocab) {
  std::vector<int> t(batch);
  for (int r = 0; r < batch; ++r) t[r] = (int)(((long long)r * 7919 + 1) % vocab);
  return t;
}

// The best of several timed runs after an untimed one: --reps runs at least, more for short
// work (up to about half a second of runs, at most 50), fewer for long steps.
static pmpp::Stat measure(const pmpp::Args& a, const std::function<pmpp::Timing()>& once) {
  const double probe = once().best();
  int reps = a.reps;
  if (probe > 2.0) reps = 1;
  else if (probe > 0.25) reps = 2;
  else if (probe > 0) reps = std::max(a.reps, std::min(50, (int)(0.5 / probe)));
  return pmpp::time_it(reps, once);
}

static int bench(const pmpp::Args& a) {
  const int vocab = 32000, layers = 2;
  int d = 4096;
  auto weightBytes = [&](int dd) { return 4.0 * ((double)vocab * dd + layers * ((double)dd * dd + dd)); };
  // Sized from what the device reports: the weights may take at most a quarter of it. Then
  // d halves until the allocations succeed.
  const double mem = deviceMemory();
  while (d > 256 && mem > 0 && weightBytes(d) > mem / 4) d /= 2;
  std::unique_ptr<DevModel> dm;
  for (;;) {
    dm.reset(new DevModel(d, vocab, layers));
    if (dm->ok) break;
    dm.reset();
    std::printf("could not allocate the weights at d = %d\n", d);
    if (d <= 256) return 1;
    d /= 2;
  }
  uploadRandom(dm->E, (size_t)vocab * d, 7, 1.0f);
  for (int l = 0; l < layers; ++l) {
    uploadRandom(dm->W[l], (size_t)d * d, 17 + l, weightScale(d));
    uploadRandom(dm->b[l], (size_t)d, 27 + l, kBiasScale);
  }
  const double wb = weightBytes(d);
  std::printf("model: d = %d, layers = %d, vocab = %d, float32, embedding tied to the LM head (shape ours)\n"
              "       weights %.1f MB; no attention (chapter 20); timing: %s\n",
              d, layers, vocab, wb / 1e6, stepUnit());

  std::map<Gemm, std::map<int, double>> stepSeconds;   // gemm -> batch -> seconds
  std::map<Gemm, int> largest;

  // ---- 'step' rows: one step for `batch` sequences
  for (Gemm g : kGemms) {
    std::string why;
    if (!gemmPresent(g, &why)) {
      std::printf("\nno %s rows: %s\n", gemmName(g), why.c_str());
      continue;
    }
    Runner run(*dm, g);
    if (!run.problem.empty()) {
      std::printf("\nno %s rows: %s\n", gemmName(g), run.problem.c_str());
      continue;
    }
    std::printf("\n%s (%s): one greedy decode step for the whole batch\n", gemmName(g), gemmSource(g));
    std::printf("  batch    step ms     tokens/s   vs half the batch   weights GB/s   model FLOP/B\n");
    double prev = 0;
    for (int B = 1; B <= 1024 && gemmTakes(g, B); B *= 2) {
      DevState s(B, d, vocab, benchTokens(B, vocab), false);
      if (!s.ok) {
        std::printf("  batch %d: allocation failed; stopping\n", B);
        pmpp::take_error();
        break;
      }
      bool ok = true;
      pmpp::Stat st = measure(a, [&] { return run.step(s, &ok); });
      std::string err = pmpp::take_error();
      if (!err.empty() || !ok || st.min_s <= 0) {
        std::printf("  batch %d: the step failed (%s); stopping\n", B, err.empty() ? "a piece refused to dispatch" : err.c_str());
        break;
      }
      const double sec = st.min_s, tps = B / sec;
      const double fpb = modelFlops(B, d, vocab, layers) / modelBytes(B, d, vocab, layers);
      char ratio[32] = "";
      if (prev > 0) std::snprintf(ratio, sizeof ratio, "%.2fx", tps / prev);
      std::printf("  %5d  %9.3f  %11.1f   %17s   %12.1f   %12.2f\n", B, sec * 1e3, tps, ratio, wb / sec / 1e9, fpb);
      pmpp::bench_row(a, "step").s("gemm", gemmName(g)).n("batch", B).n("seconds", sec).n("median_s", st.median_s)
          .n("tokens_per_s", tps).n("weight_bytes", wb).n("weight_gb_per_s", wb / sec / 1e9)
          .n("model_flop_per_byte", fpb).n("d", d).n("layers", layers).n("vocab", vocab).emit();
      stepSeconds[g][B] = sec;
      largest[g] = B;
      prev = tps;
      if (sec > 2.0) {
        std::printf("  a step took more than 2 s: stopping\n");
        break;
      }
    }
  }

  // ---- 'one_by_one' rows: that many separate batch-1 steps, in one command buffer / event pair
  const int kSolo = 32;
  for (Gemm g : kGemms) {
    if (!largest.count(g)) continue;
    Runner run(*dm, g);
    std::vector<std::unique_ptr<DevState>> solo;
    for (int i = 0; i < kSolo; ++i) {
      solo.emplace_back(new DevState(1, d, vocab, {(int)(((long long)i * 7919 + 1) % vocab)}, false));
      if (!solo.back()->ok) break;
    }
    if ((int)solo.size() < kSolo || !solo.back()->ok) {
      std::printf("\n%s one by one: allocation failed\n", gemmName(g));
      pmpp::take_error();
      continue;
    }
    std::printf("\n%s: the same number of sequences, each decoded alone (batch-1 steps one after another)\n",
                gemmName(g));
    std::printf("  sequences   one by one ms   tokens/s   batched step ms   tokens/s\n");
    for (int B = 1; B <= kSolo; B *= 2) {
      bool ok = true;
      pmpp::Stat st = measure(a, [&] { return run.steps(solo, B, &ok); });
      std::string err = pmpp::take_error();
      if (!err.empty() || !ok || st.min_s <= 0) {
        std::printf("  %d: failed (%s)\n", B, err.c_str());
        break;
      }
      const double sec = st.min_s;
      const auto it = stepSeconds[g].find(B);
      if (it != stepSeconds[g].end()) {
        std::printf("  %9d   %13.3f   %8.1f   %15.3f   %8.1f\n", B, sec * 1e3, B / sec, it->second * 1e3, B / it->second);
      } else {
        std::printf("  %9d   %13.3f   %8.1f\n", B, sec * 1e3, B / sec);
      }
      pmpp::bench_row(a, "one_by_one").s("gemm", gemmName(g)).n("batch", B).n("seconds", sec)
          .n("tokens_per_s", B / sec).emit();
    }
  }

  // ---- 'piece' rows: each piece of the step timed alone
  for (Gemm g : kGemms) {
    if (!largest.count(g)) continue;
    Runner run(*dm, g);
    std::vector<int> bs;
    for (int B : {1, 16, 256, largest[g]})
      if (B <= largest[g] && std::find(bs.begin(), bs.end(), B) == bs.end()) bs.push_back(B);
    std::printf("\n%s: each piece alone, ms (a step runs linear, bias and residual %d times each)\n", gemmName(g), layers);
    std::printf("  batch");
    for (int p = 0; p < NUM_PIECES; ++p) std::printf("  %9s", kPieceNames[p]);
    std::printf("   one step\n");
    for (int B : bs) {
      DevState s(B, d, vocab, benchTokens(B, vocab), false);
      if (!s.ok) {
        pmpp::take_error();
        std::printf("  %5d  allocation failed\n", B);
        continue;
      }
      // Measure the row first, then print it, then emit its machine-readable lines: an
      // "@pmpp" line must start a line of its own.
      std::vector<double> sec;
      std::string err;
      for (int p = 0; p < NUM_PIECES && err.empty(); ++p) {
        pmpp::Stat st = measure(a, [&] { return run.piece((Piece)p, s); });
        err = pmpp::take_error();
        if (err.empty()) sec.push_back(st.min_s);
      }
      std::printf("  %5d", B);
      for (double x : sec) std::printf("  %9.3f", x * 1e3);
      if (!err.empty()) std::printf("  failed (%s)", err.c_str());
      const auto it = stepSeconds[g].find(B);
      if (err.empty() && it != stepSeconds[g].end()) std::printf("   %8.3f", it->second * 1e3);
      std::printf("\n");
      for (size_t p = 0; p < sec.size(); ++p) {
        pmpp::bench_row(a, "piece").s("gemm", gemmName(g)).n("batch", B).s("piece", kPieceNames[p])
            .n("seconds", sec[p]).emit();
      }
    }
  }
  std::printf("\nThese are measurements of this machine. The knee (the first batch whose time per step is at\n"
              "least 1.5x the time at the smallest batch) is computed in the app from the 'step' rows.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
