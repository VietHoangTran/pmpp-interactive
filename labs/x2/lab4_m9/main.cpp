// Lab X2.4 driver (milestone M9). Tests your quantised greedy decode step against a CPU
// reference run in double on the dequantised weights, then measures it against the float32
// step at batches 1 to 8. You do not need to edit this file; your work goes in work/.
//
//   qdecode.h                  the quantised model, your entry points, the provided pieces
//   ../shared-g12/qformats.h   the block formats and llama.cpp's CPU reference quantisers
//   provided/pieces.cmake      copies the earlier labs' reference solutions into the build
//   provided/step_timer.cu     one pair of events around a stretch of launches (CUDA, HIP)
//   makeModel, quantizeMatrix  the deterministic toy model and its quantisation at load (below)
//   cpuDecodeStep              the CPU reference, in double (below)
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "qdecode.h"
#include "pmpp/harness.hpp"
using pmpp::Buf;

#ifndef PMPP_Q8_0_FROM
#define PMPP_Q8_0_FROM "unknown"
#endif
#ifndef PMPP_Q4_0_FROM
#define PMPP_Q4_0_FROM "unknown"
#endif

// ---------------------------------------------------------------- formats and format pairs
static const char* fmtName(QFormat f) { return f == QF_F32 ? "f32" : f == QF_Q8_0 ? "q8_0" : "q4_0"; }
// Bytes of a rows x cols matrix in format f (x2-facts E1).
static double matrixBytes(QFormat f, double rows, double cols) {
  return f == QF_F32 ? 4.0 * rows * cols : f == QF_Q8_0 ? rows * cols * (1.0 + 2.0 / 32.0) : rows * cols * 18.0 / 32.0;
}

struct Pair {
  QFormat layers, head;   // the format of every W_l, and of E (the embedding and the LM head)
};
static const Pair kPairs[] = {{QF_F32, QF_F32}, {QF_Q8_0, QF_Q8_0}, {QF_Q4_0, QF_Q4_0}, {QF_Q4_0, QF_F32}};
static std::string pairName(const Pair& p) { return std::string(fmtName(p.layers)) + "/" + fmtName(p.head); }

static const float kTemperature = 0.8f;   // softmax scale 1/0.8: a wrong scale cannot pass for a right one

// ---------------------------------------------------------------- the model, on the host
struct HostModel {
  int d = 0, vocab = 0, layers = 0;
  std::vector<float> E;                      // [vocab x d]
  std::vector<std::vector<float>> W, b;      // W[l]: [d x d] (out, in); b[l]: [d]
};

// Lab 22.2's generator: E uniform in [-1, 1]; W scaled so that a layer changes H by a fraction
// of itself; small biases. Deterministic, no downloads.
static const float kBiasScale = 0.1f;
static float weightScale(int d) { return 0.5f / std::sqrt((float)d); }
static HostModel makeModel(int d, int vocab, int layers, uint32_t seedE, uint32_t seedW, uint32_t seedB) {
  HostModel m;
  m.d = d;
  m.vocab = vocab;
  m.layers = layers;
  m.E = pmpp::random_floats((size_t)vocab * d, seedE);
  for (int l = 0; l < layers; ++l) {
    std::vector<float> w = pmpp::random_floats((size_t)d * d, seedW + l);
    for (float& x : w) x *= weightScale(d);
    std::vector<float> bb = pmpp::random_floats((size_t)d, seedB + l);
    for (float& x : bb) x *= kBiasScale;
    m.W.push_back(std::move(w));
    m.b.push_back(std::move(bb));
  }
  return m;
}

// One weight matrix in one format, quantised at load with the CPU reference quantisers
// (llama.cpp's _ref rules, qformats.h), and its dequantised floats for the CPU reference.
struct HostQ {
  QFormat fmt = QF_F32;
  int rows = 0, cols = 0;
  std::vector<float> f32;
  std::vector<int8_t> q8;
  std::vector<uint16_t> q8d;
  std::vector<BlockQ4_0> q4;
  std::vector<float> deq;   // empty unless asked for
};
static HostQ quantizeMatrix(const std::vector<float>& w, int rows, int cols, QFormat f, bool keepDeq) {
  HostQ q;
  q.fmt = f;
  q.rows = rows;
  q.cols = cols;
  const size_t n = (size_t)rows * cols;
  if (keepDeq) q.deq.resize(n);
  if (f == QF_F32) {
    q.f32 = w;
    if (keepDeq) q.deq = w;
  } else if (f == QF_Q8_0) {
    q.q8.resize(n);
    q.q8d.resize(n / QK8_0);
    for (int r = 0; r < rows; ++r) {
      qref::quantizeQ8_0Row(&w[(size_t)r * cols], &q.q8[(size_t)r * cols], &q.q8d[(size_t)r * cols / QK8_0], cols);
      if (keepDeq)
        qref::dequantizeQ8_0Row(&q.q8[(size_t)r * cols], &q.q8d[(size_t)r * cols / QK8_0], &q.deq[(size_t)r * cols], cols);
    }
  } else {
    q.q4.resize(n / QK4_0);
    for (int r = 0; r < rows; ++r) {
      qref::quantizeQ4_0Row(&w[(size_t)r * cols], &q.q4[(size_t)r * cols / QK4_0], cols);
      if (keepDeq) qref::dequantizeQ4_0Row(&q.q4[(size_t)r * cols / QK4_0], &q.deq[(size_t)r * cols], cols);
    }
  }
  return q;
}

// ---------------------------------------------------------------- device buffers
// Every output buffer is followed by a guard band of kGuard 4-byte words holding kSentinel;
// outputs start as NaN. The weights are allocated to their exact size, so that on the emulator
// a read past one is an AddressSanitizer error.
static const int kGuard = 64;
static const uint32_t kSentinel = 0x5EEDF00Du;
static const uint32_t kQuietNaN = 0x7FC00000u;

struct DevQ {
  QFormat fmt = QF_F32;
  int rows = 0, cols = 0;
  Buf *f32 = nullptr, *q8 = nullptr, *q8d = nullptr, *q4 = nullptr;
  DevQ() {}
  DevQ(const DevQ&) = delete;
  ~DevQ() {
    for (Buf* x : {f32, q8, q8d, q4}) pmpp::release(x);
  }
  void upload(const HostQ& h) {
    fmt = h.fmt;
    rows = h.rows;
    cols = h.cols;
    if (h.fmt == QF_F32) {
      f32 = pmpp::alloc(h.f32.size() * 4);
      pmpp::upload(f32, h.f32.data(), h.f32.size() * 4);
    } else if (h.fmt == QF_Q8_0) {
      q8 = pmpp::alloc(h.q8.size());
      q8d = pmpp::alloc(h.q8d.size() * 2);
      pmpp::upload(q8, h.q8.data(), h.q8.size());
      pmpp::upload(q8d, h.q8d.data(), h.q8d.size() * 2);
    } else {
      q4 = pmpp::alloc(h.q4.size() * sizeof(BlockQ4_0));
      pmpp::upload(q4, h.q4.data(), h.q4.size() * sizeof(BlockQ4_0));
    }
  }
  // Empty if the device copy still holds exactly what was uploaded.
  std::string unchanged(const HostQ& h, const std::string& what) const {
    auto same = [](Buf* b, const void* want, size_t n) {
      std::vector<unsigned char> got(n);
      pmpp::download(b, got.data(), n);
      return std::memcmp(got.data(), want, n) == 0;
    };
    bool ok = h.fmt == QF_F32   ? same(f32, h.f32.data(), h.f32.size() * 4)
              : h.fmt == QF_Q8_0 ? same(q8, h.q8.data(), h.q8.size()) && same(q8d, h.q8d.data(), h.q8d.size() * 2)
                                 : same(q4, h.q4.data(), h.q4.size() * sizeof(BlockQ4_0));
    return ok ? "" : what + " (" + fmtName(h.fmt) + ") was changed";
  }
};

// The model in every format it is run in: host copies (quantised, and dequantised for the
// reference when asked) and device copies, built on first use.
struct Store {
  HostModel h;
  bool keepDeq = false;
  std::map<int, HostQ> hE;                          // format -> E
  std::map<int, std::vector<HostQ>> hW;             // format -> W[l]
  std::map<int, std::unique_ptr<DevQ>> dE;
  std::map<int, std::vector<std::unique_ptr<DevQ>>> dW;
  std::vector<Buf*> db;
  bool ok = true;
  Store() {}
  Store(const Store&) = delete;
  ~Store() {
    for (Buf* x : db) pmpp::release(x);
  }
  void uploadBiases() {
    for (int l = 0; l < h.layers; ++l) {
      db.push_back(pmpp::alloc((size_t)h.d * 4));
      pmpp::upload(db.back(), h.b[l].data(), (size_t)h.d * 4);
    }
  }
  void need(QFormat f, bool asHead, bool asLayers) {
    pmpp::take_error();
    if (asHead && !hE.count(f)) {
      hE[f] = quantizeMatrix(h.E, h.vocab, h.d, f, keepDeq);
      dE[f].reset(new DevQ());
      dE[f]->upload(hE[f]);
    }
    if (asLayers && !hW.count(f)) {
      for (int l = 0; l < h.layers; ++l) {
        hW[f].push_back(quantizeMatrix(h.W[l], h.d, h.d, f, keepDeq));
        dW[f].emplace_back(new DevQ());
        dW[f].back()->upload(hW[f].back());
      }
    }
    if (!pmpp::take_error().empty()) ok = false;
  }
  void need(const Pair& p) {
    need(p.head, true, false);
    need(p.layers, false, true);
  }
  // Frees the host copies once the device has them (the benchmark: no CPU reference).
  void dropHost() {
    for (auto& kv : hE) { kv.second.f32.clear(); kv.second.q8.clear(); kv.second.q8d.clear(); kv.second.q4.clear(); }
    for (auto& kv : hW)
      for (HostQ& q : kv.second) { q.f32.clear(); q.q8.clear(); q.q8d.clear(); q.q4.clear(); }
  }
  double weightBytes(const Pair& p) const {   // what one step reads: E (the head), every W_l, the biases
    return matrixBytes(p.head, h.vocab, h.d) + h.layers * matrixBytes(p.layers, h.d, h.d) + 4.0 * h.layers * h.d;
  }
  std::string unchanged(const Pair& p) const {
    std::string e = dE.at(p.head)->unchanged(hE.at(p.head), "E");
    for (int l = 0; l < h.layers && e.empty(); ++l)
      e = dW.at(p.layers)[l]->unchanged(hW.at(p.layers)[l], "W[" + std::to_string(l) + "]");
    for (int l = 0; l < h.layers && e.empty(); ++l) {
      std::vector<float> got(h.d);
      pmpp::download(db[l], got.data(), (size_t)h.d * 4);
      if (std::memcmp(got.data(), h.b[l].data(), (size_t)h.d * 4) != 0) e = "b[" + std::to_string(l) + "] was changed";
    }
    return e;
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
  static void fill(Buf* x, size_t n, bool poison) {
    std::vector<uint32_t> w(n + kGuard, poison ? kQuietNaN : 0u);
    for (size_t i = n; i < w.size(); ++i) w[i] = kSentinel;
    pmpp::upload(x, w.data(), w.size() * 4);
  }
};

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

static std::string buildProblem(id<MTLComputePipelineState> p, const std::string& what) {
  return p ? "" : what + " did not build: " + pmpp::take_error();
}
// Lab 10.3's rule: 256 threads per threadgroup, in whole SIMD-groups, no more groups than lanes.
static std::string m4Problem(id<MTLComputePipelineState> p, const std::string& what) {
  if (!p) return what + " did not build: " + pmpp::take_error();
  const NSUInteger w = p.threadExecutionWidth;
  if (p.maxTotalThreadsPerThreadgroup < 256 || w == 0 || 256 % w != 0 || 256 / w > w)
    return what + ": cannot run 256 threads per threadgroup in whole SIMD-groups here";
  return "";
}

static QPipelines makePipelines(std::string* problem) {
  QPipelines p{};
  auto first = [&](const std::string& e) { if (problem->empty() && !e.empty()) *problem = e; };
  p.embedF32 = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "embed_rows_f32");
  first(buildProblem(p.embedF32, "your kernel.metal's embed_rows_f32"));
  p.embedQ8_0 = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "embed_rows_q8_0");
  first(buildProblem(p.embedQ8_0, "your kernel.metal's embed_rows_q8_0"));
  p.embedQ4_0 = pmpp::metal::pipeline(PMPP_METAL_SOURCE, "embed_rows_q4_0");
  first(buildProblem(p.embedQ4_0, "your kernel.metal's embed_rows_q4_0"));
  p.gemvF32 = pmpp::metal::pipeline(PMPP_GEMV_METAL, "gemv_gather");
  first(buildProblem(p.gemvF32, "lab 22.1's gemv_gather"));
  p.gemvQ8_0 = pmpp::metal::pipeline(PMPP_Q8_0_METAL, PMPP_Q8_0_KERNEL);
  first(buildProblem(p.gemvQ8_0, "the Q8_0 GEMV (" PMPP_Q8_0_FROM ")"));
  p.gemvQ4_0 = pmpp::metal::pipeline(PMPP_Q4_0_METAL, PMPP_Q4_0_KERNEL);
  first(buildProblem(p.gemvQ4_0, "the Q4_0 GEMV (" PMPP_Q4_0_FROM ")"));
  p.addBias = pmpp::metal::pipeline(PMPP_M0_METAL, "add_bias");
  first(buildProblem(p.addBias, "lab 2.3's add_bias"));
  p.addResidual = pmpp::metal::pipeline(PMPP_M0_METAL, "add_residual");
  first(buildProblem(p.addResidual, "lab 2.3's add_residual"));
  p.softmax = pmpp::metal::pipeline(PMPP_M4_METAL, "softmax_rows");
  first(m4Problem(p.softmax, "lab 10.3's softmax_rows"));
  p.argmax = pmpp::metal::pipeline(PMPP_M4_METAL, "argmax_rows");
  first(m4Problem(p.argmax, "lab 10.3's argmax_rows"));
  return p;
}

static QMatrix view(const DevQ& q) {
  QMatrix m{};
  m.fmt = q.fmt;
  m.rows = (uint32_t)q.rows;
  m.cols = (uint32_t)q.cols;
  if (q.f32) m.f32 = buffer(q.f32);
  if (q.q8) m.q8 = buffer(q.q8);
  if (q.q8d) m.q8d = buffer(q.q8d);
  if (q.q4) m.q4 = buffer(q.q4);
  return m;
}
static id<MTLComputeCommandEncoder> encoderOf(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }

// A model in one format pair, ready to run steps. Each call is one command buffer.
struct Runner {
  QDecodeModel m{};
  std::vector<QMatrix> W;
  std::string problem;
  Runner(Store& st, const Pair& pr) {
    m.pipelines = makePipelines(&problem);
    m.d = (uint32_t)st.h.d;
    m.vocab = (uint32_t)st.h.vocab;
    m.layers = (uint32_t)st.h.layers;
    m.E = view(*st.dE.at(pr.head));
    for (int l = 0; l < st.h.layers; ++l) W.push_back(view(*st.dW.at(pr.layers)[l]));
    m.W = W.data();
    NSMutableArray<id<MTLBuffer>>* B = [[NSMutableArray alloc] initWithCapacity:st.h.layers];
    for (int l = 0; l < st.h.layers; ++l) [B addObject:buffer(st.db[l])];
    m.b = B;
  }
  Runner(const Runner&) = delete;
  static DecodeState state(DevState& s) {
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
  pmpp::Timing step(DevState& s, bool* ok) {
    DecodeState v = state(s);
    *ok = true;
    return pmpp::run([&](void* pass) { *ok = encodeQDecodeStep(encoderOf(pass), m, v); });
  }
  pmpp::Timing embed(DevState& s) {
    return pmpp::run([&](void* pass) {
      encodeEmbedQ(encoderOf(pass), m.pipelines, m.E, buffer(s.tokens), buffer(s.H), (uint32_t)s.batch);
    });
  }
};
static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* stepUnit() { return "one command buffer per step, GPUEndTime - GPUStartTime"; }

#else   // CUDA, HIP, emulators
float eventTimeMs(void (*enqueue)(void* ctx), void* ctx);   // provided/step_timer.cu

template <class T> static T* ptr(Buf* b) { return b ? (T*)pmpp::raw(b) : nullptr; }
static QMatrix view(const DevQ& q) {
  QMatrix m{};
  m.fmt = q.fmt;
  m.rows = q.rows;
  m.cols = q.cols;
  m.f32 = ptr<const float>(q.f32);
  m.q8 = ptr<const int8_t>(q.q8);
  m.q8d = ptr<const uint16_t>(q.q8d);
  m.q4 = ptr<const BlockQ4_0>(q.q4);
  return m;
}
// Runs `work` between two events (provided/step_timer.cu) inside pmpp::run, which checks for
// launch errors and waits. The device's event time replaces the host's clock.
static pmpp::Timing exec(const std::function<void()>& work) {
  float ms = -1.0f;
  pmpp::Timing t = pmpp::run([&](void*) {
    ms = eventTimeMs([](void* c) { (*(const std::function<void()>*)c)(); }, (void*)&work);
  });
  if (ms >= 0) t.gpu_s = ms * 1e-3;
  return t;
}
struct Runner {
  QDecodeModel m{};
  std::vector<QMatrix> W;
  std::vector<const float*> b;
  std::string problem;
  Runner(Store& st, const Pair& pr) {
    m.d = st.h.d;
    m.vocab = st.h.vocab;
    m.layers = st.h.layers;
    m.E = view(*st.dE.at(pr.head));
    for (int l = 0; l < st.h.layers; ++l) {
      W.push_back(view(*st.dW.at(pr.layers)[l]));
      b.push_back(ptr<const float>(st.db[l]));
    }
    m.W = W.data();
    m.b = b.data();
  }
  Runner(const Runner&) = delete;
  static DecodeState state(DevState& s) {
    DecodeState v{};
    v.batch = s.batch;
    v.temperature = kTemperature;
    v.tokens = ptr<int>(s.tokens);
    v.H = ptr<float>(s.H);
    v.T = ptr<float>(s.T);
    v.logits = ptr<float>(s.logits);
    v.probs = ptr<float>(s.probs);
    return v;
  }
  pmpp::Timing step(DevState& s, bool* ok) {
    *ok = true;
    DecodeState v = state(s);
    return exec([&] { launchQDecodeStep(m, v); });
  }
  pmpp::Timing embed(DevState& s) {
    return exec([&] { launchEmbedQ(m.E, ptr<int>(s.tokens), ptr<float>(s.H), s.batch); });
  }
};
static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* stepUnit() { return "one pair of events per step"; }
#endif

// ---------------------------------------------------------------- the CPU reference
// Lab 22.2's tolerances. H and the logits: |got - ref| <= kRtol * (1 + mag), where mag is the
// sum of the absolute values of every product and bias that went into the element. The
// probabilities are checked against the softmax of the logits your step computed.
static const double kRtol = 1e-4;
static const double kProbRtol = 1e-4, kProbAtol = 1e-20;

struct RefStep {
  int next = -1;              // argmax of the logits, the lowest index on a tie (M4's rule)
  bool clear = false;         // the top logit beats the runner-up by more than both tolerances
  std::vector<double> H, magH, logits, magL;
};

// One greedy step for one sequence on `token`, in double, on the dequantised weights E and W
// (the floats the CPU's dequantisers give; for F32 the weights themselves). With no attention,
// the step depends on the current token alone, so it is computed once per token.
static void cpuDecodeStep(const HostModel& m, const std::vector<float>& E, const std::vector<const std::vector<float>*>& W,
                          int token, RefStep& out) {
  const int d = m.d;
  out.H.assign(d, 0.0);
  out.magH.assign(d, 0.0);
  for (int j = 0; j < d; ++j) {
    out.H[j] = E[(size_t)token * d + j];
    out.magH[j] = std::fabs(out.H[j]);
  }
  std::vector<double> T(d), magT(d);
  for (int l = 0; l < m.layers; ++l) {
    const float* w = W[l]->data();
    for (int o = 0; o < d; ++o) {
      double acc = 0.0, mag = 0.0;
      for (int k = 0; k < d; ++k) {
        const double x = w[(size_t)o * d + k];
        acc += out.H[k] * x;
        mag += out.magH[k] * std::fabs(x);
      }
      T[o] = acc + m.b[l][o];
      magT[o] = mag + std::fabs((double)m.b[l][o]);
    }
    for (int j = 0; j < d; ++j) {
      out.H[j] += T[j];
      out.magH[j] += magT[j];
    }
  }
  out.logits.assign(m.vocab, 0.0);
  out.magL.assign(m.vocab, 0.0);
  for (int v = 0; v < m.vocab; ++v) {
    const float* e = &E[(size_t)v * d];
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
    if (out.logits[v] > out.logits[best]) best = v;
  int second = best == 0 ? 1 : 0;
  for (int v = 0; v < m.vocab; ++v)
    if (v != best && out.logits[v] > out.logits[second]) second = v;
  out.next = best;
  // A correct step's logits are each within kRtol (1 + mag) of these, so its argmax must be
  // `best` whenever the gap exceeds the two tolerances together.
  out.clear = m.vocab < 2 || out.logits[best] - out.logits[second] > kRtol * (2.0 + out.magL[best] + out.magL[second]);
}

// The reference for one format pair, computed on demand per token.
struct Reference {
  Store* st = nullptr;
  Pair pair{};
  std::map<int, RefStep> byToken;
  const RefStep& at(int token) {
    auto it = byToken.find(token);
    if (it != byToken.end()) return it->second;
    RefStep& r = byToken[token];
    std::vector<const std::vector<float>*> W;
    for (int l = 0; l < st->h.layers; ++l) W.push_back(&st->hW.at(pair.layers)[l].deq);
    cpuDecodeStep(st->h, st->hE.at(pair.head).deq, W, token, r);
    return r;
  }
};

// The first tokens of a batch: 0 and vocab - 1 (the ends of the embedding), then spread out.
static std::vector<int> startTokens(int batch, int vocab) {
  std::vector<int> t(batch);
  for (int r = 0; r < batch; ++r) t[r] = r == 0 ? 0 : r == 1 ? vocab - 1 : (int)(((long long)r * 7919 + 11) % vocab);
  return t;
}

// ---------------------------------------------------------------- tests
static std::string at(int step) { return "after step " + std::to_string(step) + ": "; }

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

// How many next tokens the decode checks compared, of how many they saw (printed per shape).
static long g_tokensCompared = 0, g_tokensSeen = 0;

// After one step: H, logits and probs of every sequence against the reference for the token it
// had before the step; its next token wherever the reference's margin is clear.
static std::string checkStep(Reference& ref, DevState& s, const std::vector<int>& before, int step, int* compared) {
  const HostModel& h = ref.st->h;
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
  const double scale = (double)(1.0f / kTemperature);
  for (int r = 0; r < B; ++r) {
    const RefStep& rs = ref.at(before[r]);
    for (int j = 0; j < d; ++j) {
      const double got = H[(size_t)r * d + j];
      if (!(std::fabs(got - rs.H[j]) <= kRtol * (1.0 + rs.magH[j]))) note(bh, r, j, got, rs.H[j]);
    }
    for (int v = 0; v < V; ++v) {
      const double got = L[(size_t)r * V + v];
      if (!(std::fabs(got - rs.logits[v]) <= kRtol * (1.0 + rs.magL[v]))) note(bl, r, v, got, rs.logits[v]);
    }
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
  if (bp.n)
    return at(step) + describeElement("probs", bp.r, bp.c, "token", bp.got, bp.want, bp.n) +
           " (expected: the softmax of your logits times 1/temperature)";
  for (int r = 0; r < B; ++r) {
    const RefStep& rs = ref.at(before[r]);
    if (toks[r] < 0 || toks[r] >= V) return at(step) + "sequence " + std::to_string(r) + " got token " + std::to_string(toks[r]);
    ++g_tokensSeen;
    if (!rs.clear) continue;   // the two best logits are within tolerance: either token is right
    ++*compared;
    ++g_tokensCompared;
    if (toks[r] != rs.next) {
      char b[300];
      std::snprintf(b, sizeof b, "sequence %d was on token %d; it got next token %d, the reference's argmax is %d", r,
                    before[r], toks[r], rs.next);
      return at(step) + b;
    }
  }
  return "";
}

// `steps` greedy steps of a batch, every one checked, then the weights. `tokens` receives the
// device's tokens after each step (the batch-invariance check reuses them).
static std::string checkDecode(Store& st, const Pair& pr, Reference& ref, int batch, int steps,
                               std::vector<std::vector<int>>* tokens) {
  tokens->clear();
  Runner run(st, pr);
  if (!run.problem.empty()) return run.problem;
  const HostModel& h = st.h;
  std::vector<int> tok = startTokens(batch, h.vocab);
  DevState s(batch, h.d, h.vocab, tok, true);
  if (!s.ok) return "could not allocate the state";
  int compared = 0;
  for (int step = 1; step <= steps; ++step) {
    bool ok = true;
    run.step(s, &ok);
    std::string err = pmpp::take_error();
    if (!err.empty()) return at(step) + err;
    if (!ok) return at(step) + "the step returned false: a piece refused to dispatch";
    std::string e = checkStep(ref, s, tok, step, &compared);
    if (!e.empty()) return e;
    std::vector<int> t = fetch<int>(s.tokens, batch);
    t.resize(batch);
    tok = t;   // the next step starts from the tokens the device chose
    tokens->push_back(tok);
  }
  return st.unchanged(pr);
}

static std::string decodeTokens(Runner& run, const HostModel& h, const std::vector<int>& first, int steps,
                                std::vector<std::vector<int>>& out) {
  DevState s((int)first.size(), h.d, h.vocab, first, true);
  if (!s.ok) return "could not allocate the state";
  for (int step = 1; step <= steps; ++step) {
    bool ok = true;
    run.step(s, &ok);
    std::string err = pmpp::take_error();
    if (!err.empty()) return at(step) + err;
    if (!ok) return at(step) + "the step returned false: a piece refused to dispatch";
    std::vector<int> t = fetch<int>(s.tokens, first.size());
    t.resize(first.size());
    out.push_back(t);
  }
  return "";
}

// Batch invariance: every sequence of a batch decodes exactly as it does alone. `together` is
// the batch's tokens from its decode check, which must have passed.
static std::string checkInvariance(Store& st, const Pair& pr, int batch, int steps,
                                   const std::vector<std::vector<int>>& together) {
  if ((int)together.size() != steps) return "not checked: this batch's decode check above did not pass";
  Runner run(st, pr);
  if (!run.problem.empty()) return run.problem;
  const std::vector<int> first = startTokens(batch, st.h.vocab);
  for (int r = 0; r < batch; ++r) {
    std::vector<std::vector<int>> alone;
    std::string e = decodeTokens(run, st.h, {first[r]}, steps, alone);
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

// The embedding alone: every row of H bit for bit the CPU's dequantised row of E.
static std::string checkEmbed(Store& st, QFormat f, const std::vector<int>& tokens) {
  const HostModel& h = st.h;
  const int B = (int)tokens.size();
  Runner run(st, Pair{f, f});
#if defined(PMPP_BACKEND_METAL)
  if (!run.m.pipelines.embedF32 || !run.m.pipelines.embedQ8_0 || !run.m.pipelines.embedQ4_0) return run.problem;
#endif
  DevState s(B, h.d, h.vocab, tokens, true);
  if (!s.ok) return "could not allocate the state";
  run.embed(s);
  std::string err = pmpp::take_error();
  if (!err.empty()) return err;
  std::vector<float> H = fetch<float>(s.H, (size_t)B * h.d);
  std::vector<int> t = fetch<int>(s.tokens, B);
  const std::vector<float>& deq = st.hE.at(f).deq;
  for (int r = 0; r < B; ++r) {
    for (int j = 0; j < h.d; ++j) {
      const float want = deq[(size_t)tokens[r] * h.d + j];
      const float got = H[(size_t)r * h.d + j];
      if (std::memcmp(&got, &want, 4) != 0) {
        char b[300];
        std::snprintf(b, sizeof b, "H[%d][%d] is %.9g; the dequantised row %d of E has %.9g there", r, j, got, tokens[r],
                      want);
        return b;
      }
    }
    if (t[r] != tokens[r]) return "the embedding changed tokens[" + std::to_string(r) + "]";
  }
  std::string g = guardsIntact(s);
  if (!g.empty()) return g;
  return st.dE.at(f)->unchanged(st.hE.at(f), "E");
}

static void testShape(pmpp::Tests& t, int d, int vocab, int layers, const std::vector<int>& batches, int steps,
                      bool invariance, const std::string& label) {
  Store st;
  st.keepDeq = true;
  st.h = makeModel(d, vocab, layers, 2201, 2302, 2402);
  st.uploadBiases();
  for (const Pair& p : kPairs) st.need(p);
  std::printf("%s: d = %d, vocab = %d, layers = %d; temperature %.2f\n", label.c_str(), d, vocab, layers,
              (double)kTemperature);
  if (!st.ok) {
    t.check(label + ": allocate the model", [] { return std::string("allocation failed"); });
    return;
  }
  if (invariance) {   // the embedding alone, once per E format, at the small shape
    const std::vector<int> embedA = {0, vocab - 1, 13, 13, 5, 1, vocab - 1};
    for (QFormat f : {QF_F32, QF_Q8_0, QF_Q4_0}) {
      t.check(std::string("embed alone, E in ") + fmtName(f) + ": batch 7, tokens 0, V-1, 13, 13, 5, 1, V-1 (bit for bit)",
              [&] { return checkEmbed(st, f, embedA); });
    }
  }
  g_tokensCompared = g_tokensSeen = 0;
  for (const Pair& p : kPairs) {
    Reference ref;
    ref.st = &st;
    ref.pair = p;
    std::map<int, std::vector<std::vector<int>>> together;
    for (int B : batches) {
      t.check(label + ", " + pairName(p) + " (layers/head), batch " + std::to_string(B) + ": " + std::to_string(steps) +
                  " steps against the CPU reference",
              [&] { return checkDecode(st, p, ref, B, steps, &together[B]); });
    }
    const int inv = batches.back();
    if (invariance && inv > 1) {
      t.check(label + ", " + pairName(p) + ", batch " + std::to_string(inv) + ": each sequence decodes as it does alone",
              [&] { return checkInvariance(st, p, inv, steps, together[inv]); });
    }
  }
  std::printf("  (%s: the next token was compared with the reference's argmax in %ld of %ld cases; in the others\n"
              "   the reference's two best logits were closer than the tolerance, so either token is correct)\n",
              label.c_str(), g_tokensCompared, g_tokensSeen);
}

static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  std::printf("Q8_0 GEMV: %s\nQ4_0 GEMV: %s\n", PMPP_Q8_0_FROM, PMPP_Q4_0_FROM);
  const bool emu = pmpp::is_emu();
  // The emulators run every thread of every launch under three thread orders: batches 1, 3 and 8
  // there keep the run to minutes.
  const std::vector<int> batches = emu ? std::vector<int>{1, 3, 8} : std::vector<int>{1, 2, 3, 4, 5, 6, 7, 8};
  testShape(t, 64, 96, 2, batches, emu ? 3 : 4, true, "toy model");
  // On a GPU, also lab 22.2's default shape: one batch of 8, two steps (the CPU reference in
  // double is what takes the time).
  if (!emu) testShape(t, 4096, 32000, 2, {8}, 2, false, "lab 22.2's shape");
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
static pmpp::Stat measure(const pmpp::Args& a, const std::function<pmpp::Timing()>& once) {
  const double probe = once().best();
  int reps = a.reps;
  if (probe > 2.0) reps = 1;
  else if (probe > 0.25) reps = 2;
  else if (probe > 0) reps = std::max(a.reps, std::min(50, (int)(0.5 / probe)));
  return pmpp::time_it(reps, once);
}

static std::vector<int> benchTokens(int batch, int vocab) {
  std::vector<int> t(batch);
  for (int r = 0; r < batch; ++r) t[r] = (int)(((long long)r * 7919 + 1) % vocab);
  return t;
}

// `steps` greedy steps from `first`; the tokens after each, and the logits after the first.
static bool trajectory(Runner& run, const HostModel& h, const std::vector<int>& first, int steps,
                       std::vector<std::vector<int>>& toks, std::vector<float>& logits1) {
  DevState s((int)first.size(), h.d, h.vocab, first, false);
  if (!s.ok) return false;
  for (int step = 0; step < steps; ++step) {
    bool ok = true;
    run.step(s, &ok);
    if (!ok || !pmpp::take_error().empty()) return false;
    if (step == 0) {
      logits1 = fetch<float>(s.logits, (size_t)s.batch * h.vocab);
      logits1.resize((size_t)s.batch * h.vocab);
    }
    std::vector<int> t = fetch<int>(s.tokens, first.size());
    t.resize(first.size());
    toks.push_back(t);
  }
  return true;
}

static int bench(const pmpp::Args& a) {
  const int vocab = 32000, layers = 2, kAgreeSteps = 32;
  int d = 4096;
  // Every format is on the device at once: F32, Q8_0 and Q4_0 copies of E and of each W_l.
  auto allBytes = [&](int dd) {
    double s = 0;
    for (QFormat f : {QF_F32, QF_Q8_0, QF_Q4_0}) s += matrixBytes(f, vocab, dd) + layers * matrixBytes(f, dd, dd);
    return s;
  };
  // Sized from what the device reports: the weights may take at most a quarter of it.
  const double mem = deviceMemory();
  while (d > 256 && mem > 0 && allBytes(d) > mem / 4) d /= 2;
  Store st;
  st.h = makeModel(d, vocab, layers, 7, 17, 27);   // lab 22.2's seeds
  st.uploadBiases();
  for (const Pair& p : kPairs) st.need(p);
  st.dropHost();
  st.h.E.clear();
  for (auto& w : st.h.W) w.clear();
  if (!st.ok) {
    std::printf("could not allocate the weights at d = %d\n", d);
    return 1;
  }
  std::printf("model: d = %d, layers = %d, vocab = %d, the embedding tied to the LM head (lab 22.2's shape and seeds)\n"
              "       quantised at load with llama.cpp's _ref rules; timing: %s\n"
              "       Q8_0 GEMV: %s\n       Q4_0 GEMV: %s\n",
              d, layers, vocab, stepUnit(), PMPP_Q8_0_FROM, PMPP_Q4_0_FROM);

  // The F32 step first: its time is the speed-up's denominator, its tokens and logits the
  // reference the quantised steps are compared with.
  std::map<int, double> f32Seconds;
  std::map<int, std::vector<std::vector<int>>> f32Tokens;
  std::map<int, std::vector<float>> f32Logits;
  std::printf("\n  layers/head  batch    step ms    tokens/s   weights MB   weights GB/s   vs f32   logits rel RMS   "
              "agree (of %d)\n", kAgreeSteps);
  for (const Pair& p : kPairs) {
    Runner run(st, p);
    if (!run.problem.empty()) {
      std::printf("  %s: not measured (%s)\n", pairName(p).c_str(), run.problem.c_str());
      continue;
    }
    const double wb = st.weightBytes(p);
    for (int B = 1; B <= QDECODE_MAX_BATCH; B *= 2) {
      DevState s(B, d, vocab, benchTokens(B, vocab), false);
      if (!s.ok) {
        std::printf("  %s batch %d: allocation failed\n", pairName(p).c_str(), B);
        pmpp::take_error();
        break;
      }
      bool ok = true;
      const pmpp::Stat stt = measure(a, [&] { return run.step(s, &ok); });
      std::string err = pmpp::take_error();
      if (!err.empty() || !ok || stt.min_s <= 0) {
        std::printf("  %s batch %d: the step failed (%s)\n", pairName(p).c_str(), B,
                    err.empty() ? "a piece refused to dispatch" : err.c_str());
        break;
      }
      const double sec = stt.min_s;
      std::vector<std::vector<int>> toks;
      std::vector<float> logits1;
      if (!trajectory(run, st.h, benchTokens(B, vocab), kAgreeSteps, toks, logits1)) {
        std::printf("  %s batch %d: the 32-step run failed\n", pairName(p).c_str(), B);
        break;
      }
      if (p.layers == QF_F32 && p.head == QF_F32) {
        f32Seconds[B] = sec;
        f32Tokens[B] = toks;
        f32Logits[B] = logits1;
      }
      double rel = -1, speed = -1;
      int agree = -1;
      if (f32Logits.count(B)) {
        double num = 0, den = 0;
        for (size_t i = 0; i < logits1.size(); ++i) {
          const double e = (double)logits1[i] - f32Logits[B][i];
          num += e * e;
          den += (double)f32Logits[B][i] * f32Logits[B][i];
        }
        rel = den > 0 ? std::sqrt(num / den) : 0;
        agree = 0;
        while (agree < kAgreeSteps && toks[agree] == f32Tokens[B][agree]) ++agree;
        speed = f32Seconds[B] / sec;
      }
      std::printf("  %-11s  %5d  %9.3f  %10.1f  %11.1f  %13.1f  %6.2fx  %15.3g  %8d\n", pairName(p).c_str(), B, sec * 1e3,
                  B / sec, wb / 1e6, wb / sec / 1e9, speed, rel, agree);
      pmpp::bench_row(a, "qstep").s("format_layers", fmtName(p.layers)).s("format_head", fmtName(p.head)).n("batch", B)
          .n("seconds", sec).n("median_s", stt.median_s).n("tokens_per_s", B / sec).n("weight_bytes", wb)
          .n("weight_gb_per_s", wb / sec / 1e9).n("speedup_vs_f32", speed).n("logits_rel_rms", rel)
          .n("agree_steps", agree).n("d", d).n("layers", layers).n("vocab", vocab).emit();
    }
  }
  std::printf("\n'agree' is the number of greedy steps, from the first, in which every sequence of the batch chose the\n"
              "same token as the f32 step (%d means it never differed). This model is random: its agreement says\n"
              "nothing about a trained model's. These are measurements of this machine.\n", kAgreeSteps);
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
