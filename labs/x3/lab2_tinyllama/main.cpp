// Lab X3.2 driver (milestone M10). Loads a Llama-style model that provided/make_model.py
// wrote from a fixed seed, checks every provided piece alone, then tests your decode step
// against a CPU reference in double and measures it. You do not need to edit this file;
// your work goes in work/.
//
//   llama.h                   the model, the state, your entry points, the provided pieces
//   provided/pieces.cmake     copies the earlier labs' reference solutions into the build
//   provided/make_model.py    writes the models (llama2.c's version 0 .bin format)
//   provided/step_timer.cu    one pair of events around a stretch of launches (CUDA, HIP)
//   RefSeq (below)            the CPU reference: llama2.c's forward, in double
//
// Not in the book. The reference follows llama2.c's run.c (notes/today/x3-facts.md D1); the
// error report is llama.cpp's NMSE (D7); the greedy rule is "lowest index" (D9, M4).
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>

#include "llama.h"
#include "pmpp/harness.hpp"
using pmpp::Buf;

// ---------------------------------------------------------------- what the step can use
enum Gemm { GEMV, M2, M3 };
static const Gemm kGemms[] = {GEMV, M2, M3};
static const char* gemmName(Gemm g) { return g == GEMV ? "gemv" : g == M2 ? "m2" : "m3"; }
static const char* gemmSource(Gemm g) {
  return g == GEMV ? "lab 22.1's gather" : g == M2 ? "lab 6.4's tiled kernel" : "lab 15.6's register-tiled kernel";
}
static const int kGemvMaxBatch = 8;   // lab 22.1's MAX_BATCH
enum Attn { DECODE, GROUPED };
static const char* attnName(Attn a) { return a == DECODE ? "lab 20.4's decode kernel" : "lab 20.5's grouped kernel"; }

// ---------------------------------------------------------------- tolerances
// A logit passes if |got - ref| <= kRtol * (1 + mag), where mag = sum_j |xn_j| |E_vj| is the
// size of the terms of the LM head's dot product (xn: the reference's final normalised
// hidden state). One value for every backend; README.md says how it was chosen.
static const double kRtol = 4e-5;
// A greedy step is conclusive when the reference's top logit beats the runner-up by more
// than both their tolerances together: then no logit error the test accepts can swap them.
static double tolOf(double mag) { return kRtol * (1.0 + mag); }
// The pieces alone: the harness's rule, |got - ref| <= 1e-4 * (1 + mag).
static const double kPieceRtol = 1e-4;

// ---------------------------------------------------------------- device buffers
// Every buffer the step writes is followed by a guard band of kGuard words holding
// kSentinel; in the tests the activations, logits and caches start as NaN. The weights and
// the other inputs are allocated to their exact size, so that on the emulator a read past
// one is an AddressSanitizer error.
static const int kGuard = 64;
static const uint32_t kSentinel = 0x5EEDF00Du;
static const uint32_t kQuietNaN = 0x7FC00000u;
static const int kNoToken = -1;

static Buf* allocWords(size_t words, bool guard) { return pmpp::alloc((words + (guard ? kGuard : 0)) * 4); }
// `words` words of `value`, then the guard band.
static void fillWords(Buf* b, size_t words, uint32_t value) {
  std::vector<uint32_t> w(words + kGuard, value);
  for (size_t i = words; i < w.size(); ++i) w[i] = kSentinel;
  pmpp::upload(b, w.data(), w.size() * 4);
}
template <class T>
static void uploadGuarded(Buf* b, const std::vector<T>& data) {
  static_assert(sizeof(T) == 4, "4-byte elements");
  std::vector<uint32_t> w(data.size() + kGuard, kSentinel);
  std::memcpy(w.data(), data.data(), data.size() * 4);
  pmpp::upload(b, w.data(), w.size() * 4);
}
// n data words plus the guard band behind them.
template <class T>
static std::vector<T> fetch(Buf* b, size_t n) {
  static_assert(sizeof(T) == 4, "4-byte elements");
  std::vector<T> v(n + kGuard);
  pmpp::download(b, v.data(), v.size() * 4);
  return v;
}
template <class T>
static std::vector<T> fetchExact(Buf* b, size_t n) {
  std::vector<T> v(n);
  pmpp::download(b, v.data(), n * sizeof(T));
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
static std::string fmt(const char* f, double a = 0, double b = 0, double c = 0, double d = 0) {
  char buf[400];
  std::snprintf(buf, sizeof buf, f, a, b, c, d);
  return buf;
}

// ---------------------------------------------------------------- the model file
// llama2.c's version 0 layout (D1): the header, then the weights in its order. Our loader
// reads only files with a shared classifier (positive vocab_size) and head size 128.
struct HostLayer {
  std::vector<float> rmsAtt, wq, wk, wv, wo, rmsFfn, w1, w2, w3;
};
struct HostModel {
  LlamaConfig c{};
  std::vector<float> E, rmsFinal;
  std::vector<HostLayer> L;
  uint64_t checksum = 0;
  long long fileBytes = 0;
};
static int dqOf(const LlamaConfig& c) { return c.n_heads * LLAMA_HEAD_DIM; }
static int dkvOf(const LlamaConfig& c) { return c.n_kv_heads * LLAMA_HEAD_DIM; }
// E1: the size of the file from its header (notes/today/x3-facts.md E1).
static long long e1Bytes(const LlamaConfig& c) {
  const long long d = c.dim, h = c.hidden_dim, L = c.n_layers, V = c.vocab_size, T = c.seq_len;
  const long long head = c.n_heads > 0 ? d / c.n_heads : 0, dkv = c.n_heads > 0 ? d * c.n_kv_heads / c.n_heads : 0;
  const long long floats = V * d + L * d + L * d * d + 2 * L * d * dkv + L * d * d + L * d + 3 * L * d * h + d + T * head;
  return 28 + 4 * floats;
}
// The weights' bytes the step reads once per token at batch 1: every matrix and norm, and E
// once for the LM head (the embedding reads one row more per sequence, not counted).
static double weightBytes(const LlamaConfig& c) {
  const double d = c.dim, h = c.hidden_dim, L = c.n_layers, V = c.vocab_size, dq = dqOf(c), dkv = dkvOf(c);
  return 4.0 * (V * d + L * (2 * d + dq * d + 2 * dkv * d + d * dq + 3 * d * h) + d);
}
static uint64_t fnv1a(const void* p, size_t n, uint64_t h = 0xCBF29CE484222325ull) {
  const unsigned char* b = (const unsigned char*)p;
  for (size_t i = 0; i < n; ++i) {
    h ^= b[i];
    h *= 0x100000001B3ull;
  }
  return h;
}
static std::string hex64(uint64_t v) {
  char b[24];
  std::snprintf(b, sizeof b, "%016llx", (unsigned long long)v);
  return b;
}

static std::string loadModel(const std::string& path, HostModel& m) {
  std::FILE* f = std::fopen(path.c_str(), "rb");
  if (!f) return "cannot open " + path + " (provided/pieces.cmake writes it at configure time; is Python 3 there?)";
  std::unique_ptr<std::FILE, int (*)(std::FILE*)> closer(f, std::fclose);
  LlamaConfig& c = m.c;
  if (std::fread(&c, sizeof c, 1, f) != 1) return path + ": shorter than the 28-byte header";
  if (c.vocab_size <= 0) return path + ": vocab_size <= 0 means an unshared classifier, which this lab does not load";
  if (c.dim <= 0 || c.hidden_dim <= 0 || c.n_layers <= 0 || c.n_heads <= 0 || c.n_kv_heads <= 0 || c.seq_len <= 0)
    return path + ": the header has a size that is not positive";
  if (c.n_heads % c.n_kv_heads != 0) return path + ": n_heads is not a multiple of n_kv_heads";
  if (c.dim != c.n_heads * LLAMA_HEAD_DIM)
    return path + fmt(": head size %g, but the decode kernels of labs 20.4 and 20.5 are compiled for 128 (x3-facts E2)",
                      (double)c.dim / c.n_heads);
  std::fseek(f, 0, SEEK_END);
  m.fileBytes = std::ftell(f);
  if (m.fileBytes != e1Bytes(c))
    return path + fmt(": %.0f bytes, but the header and llama2.c's layout (E1) say %.0f", (double)m.fileBytes,
                      (double)e1Bytes(c));
  std::fseek(f, sizeof c, SEEK_SET);
  bool ok = true;
  auto read = [&](std::vector<float>& v, size_t n) {
    v.resize(n);
    ok = ok && std::fread(v.data(), 4, n, f) == n;
  };
  const size_t d = c.dim, h = c.hidden_dim, dq = dqOf(c), dkv = dkvOf(c);
  read(m.E, (size_t)c.vocab_size * d);
  m.L.assign(c.n_layers, HostLayer());
  for (auto& l : m.L) read(l.rmsAtt, d);
  for (auto& l : m.L) read(l.wq, dq * d);
  for (auto& l : m.L) read(l.wk, dkv * d);
  for (auto& l : m.L) read(l.wv, dkv * d);
  for (auto& l : m.L) read(l.wo, d * dq);
  for (auto& l : m.L) read(l.rmsFfn, d);
  for (auto& l : m.L) read(l.w1, h * d);
  for (auto& l : m.L) read(l.w2, d * h);
  for (auto& l : m.L) read(l.w3, h * d);
  read(m.rmsFinal, d);
  if (!ok) return path + ": short read";
  // The two RoPE tables llama2.c skips follow; the driver computes its own (llama.h).
  m.checksum = fnv1a(m.rmsFinal.data() + d - 256, 256 * 4, fnv1a(m.E.data(), 256 * 4));
  return "";
}
static std::string shape(const LlamaConfig& c) {
  char b[300];
  std::snprintf(b, sizeof b, "d = %d (h_q %d, h_kv %d, head 128), hidden %d, %d layers, vocab %d, seq_len %d", c.dim,
                c.n_heads, c.n_kv_heads, c.hidden_dim, c.n_layers, c.vocab_size, c.seq_len);
  return b;
}

// ---------------------------------------------------------------- the CPU reference
// llama2.c's forward (D1), in double, from the same float weights, one sequence at a time.
static const double kEps = (double)1e-5f;     // the float the step passes, in double
static double ropeFreq(int i) { return std::pow(10000.0, -2.0 * i / LLAMA_HEAD_DIM); }

struct RefStep {
  int fed = 0;                        // the token fed at this step (position = step)
  std::vector<double> logits, mag;    // the logits and the size of their terms
  int argmax = 0;                     // the lowest index of the maximum (M4's and llama2.c's rule)
  double gap = 0, need = 0;           // top minus runner-up, and the two tolerances added
  bool conclusive() const { return gap > need; }
};
struct RefTrace {
  std::vector<int> prompt;
  std::vector<RefStep> steps;
};

class RefSeq {
 public:
  RefSeq(const HostModel& m, int mode) : m_(m), mode_(mode) {
    const LlamaConfig& c = m.c;
    K_.assign(c.n_layers, std::vector<double>((size_t)c.seq_len * dkvOf(c)));
    V_ = K_;
  }
  void step(int token, int pos, RefStep& out) {
    const LlamaConfig& c = m_.c;
    const int d = c.dim, dq = dqOf(c), dkv = dkvOf(c), hd = LLAMA_HEAD_DIM, hq = c.n_heads, hkv = c.n_kv_heads;
    std::vector<double> x(d), xb(d), xb2(d), q(dq), k(dkv), v(dkv), att(dq), hb(c.hidden_dim), hb2(c.hidden_dim);
    for (int j = 0; j < d; ++j) x[j] = m_.E[(size_t)token * d + j];
    for (int l = 0; l < c.n_layers; ++l) {
      const HostLayer& w = m_.L[l];
      rmsnorm(x, w.rmsAtt, xb);
      mv(w.wq, xb, q);
      mv(w.wk, xb, k);
      mv(w.wv, xb, v);
      rope(q, hq, pos);
      rope(k, hkv, pos);
      std::copy(k.begin(), k.end(), K_[l].begin() + (size_t)pos * dkv);
      std::copy(v.begin(), v.end(), V_[l].begin() + (size_t)pos * dkv);
      const int group = hq / hkv;
      std::vector<double> s(pos + 1);
      for (int h = 0; h < hq; ++h) {
        const int kvh = h / group;
        double mx = -1e300;
        for (int t = 0; t <= pos; ++t) {
          double dot = 0;
          for (int i = 0; i < hd; ++i) dot += q[h * hd + i] * K_[l][(size_t)t * dkv + kvh * hd + i];
          s[t] = dot / std::sqrt((double)hd);
          mx = std::max(mx, s[t]);
        }
        double z = 0;
        for (int t = 0; t <= pos; ++t) z += (s[t] = std::exp(s[t] - mx));
        for (int i = 0; i < hd; ++i) {
          double o = 0;
          for (int t = 0; t <= pos; ++t) o += s[t] * V_[l][(size_t)t * dkv + kvh * hd + i];
          att[h * hd + i] = o / z;
        }
      }
      mv(w.wo, att, xb2);
      for (int j = 0; j < d; ++j) x[j] += xb2[j];
      rmsnorm(x, w.rmsFfn, xb);
      mv(w.w1, xb, hb);
      mv(w.w3, xb, hb2);
      for (size_t i = 0; i < hb.size(); ++i) hb[i] = hb[i] / (1.0 + std::exp(-hb[i])) * hb2[i];
      mv(w.w2, hb, xb2);
      for (int j = 0; j < d; ++j) x[j] += xb2[j];
    }
    rmsnorm(x, m_.rmsFinal, xb);
    const int V = c.vocab_size;
    out.fed = token;
    out.logits.assign(V, 0.0);
    out.mag.assign(V, 0.0);
    for (int t = 0; t < V; ++t) {
      const float* e = &m_.E[(size_t)t * d];
      double acc = 0, mag = 0;
      for (int j = 0; j < d; ++j) {
        acc += xb[j] * e[j];
        mag += std::fabs(xb[j] * e[j]);
      }
      out.logits[t] = acc;
      out.mag[t] = mag;
    }
    int best = 0;
    for (int t = 1; t < V; ++t)
      if (out.logits[t] > out.logits[best]) best = t;     // strictly greater: ties keep the lower index
    int second = best == 0 ? 1 : 0;
    for (int t = 0; t < V; ++t)
      if (t != best && out.logits[t] > out.logits[second]) second = t;
    out.argmax = best;
    out.gap = out.logits[best] - out.logits[second];
    out.need = tolOf(out.mag[best]) + tolOf(out.mag[second]);
  }

 private:
  void rmsnorm(const std::vector<double>& x, const std::vector<float>& w, std::vector<double>& y) const {
    double ss = 0;
    for (double a : x) ss += a * a;
    const double inv = 1.0 / std::sqrt(ss / x.size() + kEps);
    for (size_t j = 0; j < x.size(); ++j) y[j] = w[j] * (inv * x[j]);
  }
  // y = W x, W stored (out, in) row-major: llama2.c's matmul (D1).
  static void mv(const std::vector<float>& W, const std::vector<double>& x, std::vector<double>& y) {
    const size_t in = x.size();
    for (size_t o = 0; o < y.size(); ++o) {
      const float* r = &W[o * in];
      double acc = 0;
      for (size_t i = 0; i < in; ++i) acc += r[i] * x[i];
      y[o] = acc;
    }
  }
  void rope(std::vector<double>& v, int heads, int pos) const {
    const int hd = LLAMA_HEAD_DIM, half = hd / 2;
    for (int h = 0; h < heads; ++h) {
      for (int i = 0; i < half; ++i) {
        const int i0 = mode_ == LLAMA_ROPE_CONSECUTIVE ? 2 * i : i;
        const int i1 = mode_ == LLAMA_ROPE_CONSECUTIVE ? 2 * i + 1 : i + half;
        const double a = pos * ropeFreq(i), cs = std::cos(a), sn = std::sin(a);
        const double x0 = v[h * hd + i0], x1 = v[h * hd + i1];
        v[h * hd + i0] = x0 * cs - x1 * sn;
        v[h * hd + i1] = x0 * sn + x1 * cs;
      }
    }
  }
  const HostModel& m_;
  int mode_;
  std::vector<std::vector<double>> K_, V_;
};

// Greedy, from `prompt`, for `steps` steps: the prompt is fed one token per step, then each
// step feeds the previous step's argmax (llama2.c's loop, D1).
static RefTrace refRun(const HostModel& m, int mode, const std::vector<int>& prompt, int steps) {
  RefTrace r;
  r.prompt = prompt;
  RefSeq seq(m, mode);
  int token = prompt[0];
  for (int s = 0; s < steps; ++s) {
    RefStep st;
    seq.step(token, s, st);
    r.steps.push_back(std::move(st));
    token = s + 1 < (int)prompt.size() ? prompt[s + 1] : r.steps.back().argmax;
  }
  return r;
}

// Sequence b's prompt: BOS (= 1, as llama2.c starts, D1), then tokens spread over the vocabulary.
static std::vector<int> promptFor(int b, int P, int V) {
  std::vector<int> p(P);
  for (int i = 0; i < P; ++i) p[i] = i == 0 ? 1 : (int)(((long long)b * 131 + (long long)i * 977 + 17) % V);
  return p;
}

// ---------------------------------------------------------------- the model and the state on the device
enum WeightIdx { W_RMS_ATT, W_Q, W_K, W_V, W_O, W_RMS_FFN, W_1, W_2, W_3, W_COUNT };
static const std::vector<float>& hostWeight(const HostLayer& l, int i) {
  const std::vector<float>* w[W_COUNT] = {&l.rmsAtt, &l.wq, &l.wk, &l.wv, &l.wo, &l.rmsFfn, &l.w1, &l.w2, &l.w3};
  return *w[i];
}

struct DevModel {
  LlamaConfig c{};
  Buf* E = nullptr;
  Buf* rmsFinal = nullptr;
  std::vector<std::array<Buf*, W_COUNT>> L;
  bool ok = false;
  explicit DevModel(const HostModel& h) : c(h.c) {
    pmpp::take_error();
    E = pmpp::alloc(h.E.size() * 4);
    rmsFinal = pmpp::alloc(h.rmsFinal.size() * 4);
    for (const HostLayer& hl : h.L) {
      std::array<Buf*, W_COUNT> a{};
      for (int i = 0; i < W_COUNT; ++i) a[i] = pmpp::alloc(hostWeight(hl, i).size() * 4);
      L.push_back(a);
    }
    ok = pmpp::take_error().empty();
    if (!ok) return;
    pmpp::upload(E, h.E.data(), h.E.size() * 4);
    pmpp::upload(rmsFinal, h.rmsFinal.data(), h.rmsFinal.size() * 4);
    for (size_t l = 0; l < h.L.size(); ++l)
      for (int i = 0; i < W_COUNT; ++i) pmpp::upload(L[l][i], hostWeight(h.L[l], i).data(), hostWeight(h.L[l], i).size() * 4);
  }
  DevModel(const DevModel&) = delete;
  ~DevModel() {
    pmpp::release(E);
    pmpp::release(rmsFinal);
    for (auto& a : L)
      for (Buf* b : a) pmpp::release(b);
  }
};

static std::string weightsUnchanged(DevModel& dm, const HostModel& h) {
  auto same = [](Buf* b, const std::vector<float>& want) {
    std::vector<float> got(want.size());
    pmpp::download(b, got.data(), got.size() * 4);
    return std::memcmp(got.data(), want.data(), got.size() * 4) == 0;
  };
  static const char* names[W_COUNT] = {"rmsAtt", "wq", "wk", "wv", "wo", "rmsFfn", "w1", "w2", "w3"};
  if (!same(dm.E, h.E)) return "the embedding E was changed (it is an input, and the LM head's weight)";
  if (!same(dm.rmsFinal, h.rmsFinal)) return "rmsFinal was changed";
  for (size_t l = 0; l < h.L.size(); ++l)
    for (int i = 0; i < W_COUNT; ++i)
      if (!same(dm.L[l][i], hostWeight(h.L[l], i))) return "layer " + std::to_string(l) + "'s " + names[i] + " was changed";
  return "";
}

struct DevState {
  LlamaConfig c{};
  int batch = 0, promptLen = 0;
  Buf *tokens = nullptr, *pos = nullptr, *len = nullptr, *prompt = nullptr, *out = nullptr, *rngStep = nullptr;
  Buf *x = nullptr, *xb = nullptr, *xb2 = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr;
  Buf *hb = nullptr, *hb2 = nullptr, *logits = nullptr, *cosT = nullptr, *sinT = nullptr;
  std::vector<Buf*> Kc, Vc;
  std::vector<int> promptHost;
  std::vector<float> cosHost, sinHost;
  bool ok = false;
  // promptTB: [promptLen x batch], promptTB[p * batch + b]. `poison`: activations, logits and
  // caches start as NaN (tests); zeros otherwise (the benchmark).
  DevState(const LlamaConfig& c_, int batch_, const std::vector<int>& promptTB, int promptLen_, bool poison)
      : c(c_), batch(batch_), promptLen(promptLen_), promptHost(promptTB) {
    pmpp::take_error();
    const size_t B = batch, d = c.dim, h = c.hidden_dim, V = c.vocab_size, T = c.seq_len;
    tokens = allocWords(B, true);
    pos = allocWords(B, true);
    len = allocWords(B, true);
    rngStep = allocWords(2 * B, true);
    prompt = allocWords(promptTB.size(), false);
    out = allocWords(T * B, true);
    for (Buf** p : {&x, &xb, &xb2}) *p = allocWords(B * d, true);
    q = allocWords(B * dqOf(c), true);
    att = allocWords(B * dqOf(c), true);
    k = allocWords(B * dkvOf(c), true);
    v = allocWords(B * dkvOf(c), true);
    hb = allocWords(B * h, true);
    hb2 = allocWords(B * h, true);
    logits = allocWords(B * V, true);
    cosT = allocWords(T * LLAMA_HEAD_DIM / 2, false);
    sinT = allocWords(T * LLAMA_HEAD_DIM / 2, false);
    for (int l = 0; l < c.n_layers; ++l) {
      Kc.push_back(allocWords(cacheWords(), true));
      Vc.push_back(allocWords(cacheWords(), true));
    }
    ok = pmpp::take_error().empty();
    if (!ok) return;
    pmpp::upload(prompt, promptTB.data(), promptTB.size() * 4);
    // cos and sin of p * 10000^(-2i/128), computed in double and rounded once (ours; llama2.c
    // computes them in float inside its loop, D1).
    for (size_t p = 0; p < T; ++p)
      for (int i = 0; i < LLAMA_HEAD_DIM / 2; ++i) {
        cosHost.push_back((float)std::cos(p * ropeFreq(i)));
        sinHost.push_back((float)std::sin(p * ropeFreq(i)));
      }
    pmpp::upload(cosT, cosHost.data(), cosHost.size() * 4);
    pmpp::upload(sinT, sinHost.data(), sinHost.size() * 4);
    std::vector<int> first(promptTB.begin(), promptTB.begin() + B);
    setPositions(0, first);
    fillWords(out, T * B, (uint32_t)kNoToken);
    const uint32_t fill = poison ? kQuietNaN : 0u;
    for (Buf* b : {x, xb, xb2}) fillWords(b, B * d, fill);
    fillWords(q, B * dqOf(c), fill);
    fillWords(att, B * dqOf(c), fill);
    fillWords(k, B * dkvOf(c), fill);
    fillWords(v, B * dkvOf(c), fill);
    fillWords(hb, B * h, fill);
    fillWords(hb2, B * h, fill);
    fillWords(logits, B * V, fill);
    for (int l = 0; l < c.n_layers; ++l) {
      fillWords(Kc[l], cacheWords(), fill);
      fillWords(Vc[l], cacheWords(), fill);
    }
  }
  DevState(const DevState&) = delete;
  ~DevState() {
    for (Buf* b : {tokens, pos, len, prompt, out, rngStep, x, xb, xb2, q, k, v, att, hb, hb2, logits, cosT, sinT})
      pmpp::release(b);
    for (Buf* b : Kc) pmpp::release(b);
    for (Buf* b : Vc) pmpp::release(b);
  }
  size_t cacheWords() const { return (size_t)batch * c.seq_len * dkvOf(c); }
  // Every sequence at position p (len p + 1) with the given current tokens; rngStep = p.
  void setPositions(int p, const std::vector<int>& toks) {
    uploadGuarded(tokens, toks);
    uploadGuarded(pos, std::vector<int>(batch, p));
    uploadGuarded(len, std::vector<int>(batch, p + 1));
    std::vector<uint32_t> r(2 * batch, 0);
    for (int b = 0; b < batch; ++b) r[2 * b] = (uint32_t)p;
    uploadGuarded(rngStep, r);
  }
  std::string guards() {
    const size_t B = batch, d = c.dim, h = c.hidden_dim, V = c.vocab_size, T = c.seq_len;
    struct G { Buf* b; size_t n; const char* name; };
    const G gs[] = {{tokens, B, "tokens"}, {pos, B, "pos"}, {len, B, "len"}, {rngStep, 2 * B, "rngStep"},
                    {out, T * B, "out"}, {x, B * d, "x"}, {xb, B * d, "xb"}, {xb2, B * d, "xb2"},
                    {q, B * dqOf(c), "q"}, {att, B * dqOf(c), "att"}, {k, B * dkvOf(c), "k"},
                    {v, B * dkvOf(c), "v"}, {hb, B * h, "hb"}, {hb2, B * h, "hb2"}, {logits, B * V, "logits"}};
    for (const G& g : gs)
      if (!guardIntact(fetch<uint32_t>(g.b, g.n), g.n)) return std::string("something wrote past the end of ") + g.name;
    for (int l = 0; l < c.n_layers; ++l) {
      if (!guardIntact(fetch<uint32_t>(Kc[l], cacheWords()), cacheWords()))
        return "something wrote past the end of layer " + std::to_string(l) + "'s key cache";
      if (!guardIntact(fetch<uint32_t>(Vc[l], cacheWords()), cacheWords()))
        return "something wrote past the end of layer " + std::to_string(l) + "'s value cache";
    }
    if (fetchExact<int>(prompt, promptHost.size()) != promptHost) return "the prompt array was changed";
    if (fetchExact<float>(cosT, cosHost.size()) != cosHost || fetchExact<float>(sinT, sinHost.size()) != sinHost)
      return "the cos/sin tables were changed";
    return "";
  }
  // After `steps` steps from position 0: cache rows below `steps` are finite, the rest still NaN.
  std::string cacheRows(int steps) {
    const int dkv = dkvOf(c), T = c.seq_len;
    for (int l = 0; l < c.n_layers; ++l) {
      for (int which = 0; which < 2; ++which) {
        std::vector<float> w = fetch<float>(which ? Vc[l] : Kc[l], cacheWords());
        for (int b = 0; b < batch; ++b)
          for (int t = 0; t < T; ++t)
            for (int i = 0; i < dkv; ++i) {
              const float val = w[((size_t)b * T + t) * dkv + i];
              const bool written = std::isfinite(val);
              if (written != (t < steps)) {
                char msg[200];
                std::snprintf(msg, sizeof msg, "layer %d's %s cache, sequence %d, row %d: %s", l, which ? "value" : "key", b,
                              t, written ? "written, but no step had that position" : "not written by the step at that position");
                return msg;
              }
            }
      }
    }
    return "";
  }
};

// ---------------------------------------------------------------- the sampler (M5), when present
struct SamplerSettings {
  float temperature = 0.8f;
  int topK = 40;
  float topP = 0.95f, minP = 0.05f;
};

// ---------------------------------------------------------------- backends
// Each op runs one provided piece (or your advance) alone, in one command buffer (Metal) or
// one pair of events (CUDA, HIP); *err receives a pipeline problem or a refused dispatch.
#if defined(PMPP_BACKEND_METAL)
#import <Metal/Metal.h>
#include "pmpp/metal.h"
#if defined(PMPP_HAVE_M5)
#include "sampler_metal.h"
#endif
using pmpp::metal::buffer;

// The provided pieces' encoders that llama.h does not declare (renamed by pieces.cmake).
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
void encodeDecodeAttention(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> q,
                           id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len, id<MTLBuffer> out, uint32_t batch,
                           uint32_t hQ, uint32_t hKv, uint32_t maxSeq, float scale);               // lab 20.4's
#if defined(PMPP_M7G_METAL)
void encodeDecodeAttentionGrouped(id<MTLComputeCommandEncoder>, id<MTLComputePipelineState>, id<MTLBuffer> q,
                                  id<MTLBuffer> Kc, id<MTLBuffer> Vc, id<MTLBuffer> len, id<MTLBuffer> out,
                                  uint32_t batch, uint32_t hQ, uint32_t hKv, uint32_t maxSeq, float scale);  // 20.5's
#endif

static bool gemmPresent(Gemm g) {
  if (g == M2) return true;
#if defined(PMPP_M3_METAL)
  if (g == M3) return true;
#endif
#if defined(PMPP_GEMV_METAL)
  if (g == GEMV) return true;
#endif
  return false;
}
static bool attnPresent(Attn a) {
#if defined(PMPP_M7G_METAL)
  return true;
#else
  return a == DECODE;
#endif
}
static bool samplerPresent() {
#if defined(PMPP_HAVE_M5)
  return true;
#else
  return false;
#endif
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
  if (!p) return what + " did not build: " + pmpp::take_error();
  if (p.maxTotalThreadsPerThreadgroup < threads)
    return what + fmt(": this pipeline allows %.0f threads per threadgroup, the kernel needs %.0f",
                      (double)p.maxTotalThreadsPerThreadgroup, (double)threads);
  return "";
}
// Labs 20.4 and 20.5 give each lane 4 of the 128 dimensions: 32-lane SIMD-groups, 256 threads
// (their drivers check the same; Apple documents no fixed width, ch20-facts C3).
static std::string attentionProblem(id<MTLComputePipelineState> p, const std::string& what) {
  std::string e = threadsProblem(p, 256, what);
  if (!e.empty()) return e;
  if (p.threadExecutionWidth != 32)
    return what + fmt(" assumes 32-lane SIMD-groups; this pipeline's threadExecutionWidth is %.0f",
                      (double)p.threadExecutionWidth);
  return "";
}

static id<MTLComputePipelineState> P(const char* path, const char* fn,
                                     const std::vector<std::pair<std::string, std::string>>& macros = {}) {
  return pmpp::metal::pipeline(path, fn, macros);
}
static id<MTLComputePipelineState> linearPso(Gemm g, std::string* err) {
  id<MTLComputePipelineState> p = nil;
  switch (g) {
    case M2:
      p = P(PMPP_M2_METAL, "linear", {{"TILE_WIDTH", std::to_string(kM2Tile)}});
      *err = threadsProblem(p, (NSUInteger)kM2Tile * kM2Tile, "lab 6.4's linear");
      break;
    case M3: {
#if defined(PMPP_M3_METAL)
      // As lab 15.6's driver: SIMD_WIDTH is the pipeline's threadExecutionWidth (64 if it says 64, else 32).
      id<MTLComputePipelineState> probe = P(PMPP_M3_METAL, "linear_m3", {{"SIMD_WIDTH", "32"}});
      const int width = (probe && probe.threadExecutionWidth == 64) ? 64 : 32;
      p = P(PMPP_M3_METAL, "linear_m3", {{"SIMD_WIDTH", std::to_string(width)}});
      *err = threadsProblem(p, 256, "lab 15.6's linear_m3");
#endif
      break;
    }
    case GEMV:
#if defined(PMPP_GEMV_METAL)
      p = P(PMPP_GEMV_METAL, "gemv_gather");
      *err = p ? "" : "lab 22.1's gemv_gather did not build: " + pmpp::take_error();
#endif
      break;
  }
  if (!p && err->empty()) *err = std::string("no ") + gemmName(g) + " on this build";
  return err->empty() ? p : nil;
}
static EncodeLinearFn linearEncoder(Gemm g) {
  switch (g) {
    case M2: return encodeM2;
#if defined(PMPP_M3_METAL)
    case M3: return encodeLinearM3;
#endif
#if defined(PMPP_GEMV_METAL)
    case GEMV: return encodeGemvGather;
#endif
    default: return nullptr;
  }
}
static id<MTLComputePipelineState> attentionPso(Attn a, int gq, std::string* err) {
  id<MTLComputePipelineState> p = nil;
  if (a == DECODE) {
    p = P(PMPP_M7_METAL, "decode_attention");
    *err = attentionProblem(p, "lab 20.4's decode_attention");
  } else {
#if defined(PMPP_M7G_METAL)
    p = P(PMPP_M7G_METAL, "decode_attention_grouped", {{"GQ", std::to_string(gq)}});
    *err = attentionProblem(p, "lab 20.5's decode_attention_grouped");
#else
    *err = "lab 20.5's solution was not found when the lab was configured";
#endif
  }
  return err->empty() ? p : nil;
}
static EncodeAttentionFn attentionEncoder(Attn a) {
#if defined(PMPP_M7G_METAL)
  if (a == GROUPED) return encodeDecodeAttentionGrouped;
#endif
  return a == DECODE ? encodeDecodeAttention : nullptr;
}

// The pipelines every step needs, whatever the GEMM and the attention kernel.
static std::string basePipelines(LlamaPipelines& p) {
  p.embed = P(PMPP_M8_METAL, "embed_rows");
  if (!p.embed) return "lab 22.2's embed_rows did not build: " + pmpp::take_error();
  p.rmsNorm = P(PMPP_X31_METAL, "rms_norm_rows");
  if (!p.rmsNorm) return "lab X3.1's rms_norm_rows did not build: " + pmpp::take_error();
  p.rope = P(PMPP_X31_METAL, "rope_rows");
  if (!p.rope) return "lab X3.1's rope_rows did not build: " + pmpp::take_error();
  p.swiglu = P(PMPP_X31_METAL, "swiglu");
  if (!p.swiglu) return "lab X3.1's swiglu did not build: " + pmpp::take_error();
  p.kvAppend = P(PMPP_M7_METAL, "kv_append");
  if (!p.kvAppend) return "lab 20.4's kv_append did not build: " + pmpp::take_error();
  p.addResidual = P(PMPP_M0_METAL, "add_residual");
  if (!p.addResidual) return "lab 2.3's add_residual did not build: " + pmpp::take_error();
  p.argmax = P(PMPP_M4_METAL, "argmax_rows");
  if (!p.argmax) return "lab 10.3's argmax_rows did not build: " + pmpp::take_error();
  return "";
}

// Yours: advance_positions from your kernel.metal (compiled at run time, so an edit needs no rebuild).
static id<MTLComputePipelineState> advancePso(std::string* err) {
  id<MTLComputePipelineState> p = P(PMPP_METAL_SOURCE, "advance_positions");
  *err = p ? "" : "your kernel.metal did not build (advance_positions): " + pmpp::take_error();
  return p;
}
static id<MTLComputeCommandEncoder> encoderOf(void* pass) { return ((pmpp::metal::Pass*)pass)->encoder; }
static pmpp::Timing run1(const std::function<void(id<MTLComputeCommandEncoder>)>& f) {
  return pmpp::run([&](void* pass) { f(encoderOf(pass)); });
}
static LlamaPipelines& bases(std::string* err) {
  static LlamaPipelines p;
  static std::string problem = basePipelines(p);
  *err = problem;
  return p;
}

static pmpp::Timing opEmbed(Buf* E, Buf* tok, Buf* H, int B, int d, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  return run1([&](id<MTLComputeCommandEncoder> e) { encodeEmbed(e, p.embed, buffer(E), buffer(tok), buffer(H), B, d); });
}
static pmpp::Timing opRmsNorm(Buf* X, Buf* w, Buf* Y, int rows, int d, float eps, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  bool ok = true;
  pmpp::Timing t = run1([&](id<MTLComputeCommandEncoder> e) {
    ok = encodeRmsNorm(e, p.rmsNorm, buffer(X), buffer(w), buffer(Y), rows, d, eps);
  });
  if (!ok) *err = "lab X3.1's encodeRmsNorm refused to dispatch";
  return t;
}
static pmpp::Timing opLinear(Gemm g, Buf* X, Buf* W, Buf* Y, int B, int in, int out, std::string* err) {
  err->clear();
  id<MTLComputePipelineState> p = linearPso(g, err);
  if (!p) return {0, -1};
  EncodeLinearFn f = linearEncoder(g);
  return run1([&](id<MTLComputeCommandEncoder> e) { f(e, p, buffer(X), buffer(W), buffer(Y), B, in, out); });
}
static pmpp::Timing opRope(Buf* X, Buf* cs, Buf* sn, Buf* pos, int B, int heads, int hd, int mode, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  return run1([&](id<MTLComputeCommandEncoder> e) {
    encodeRope(e, p.rope, buffer(X), buffer(pos), buffer(cs), buffer(sn), B, heads, hd, mode);
  });
}
static pmpp::Timing opKvAppend(Buf* k, Buf* v, Buf* Kc, Buf* Vc, Buf* pos, int B, int T, int hkv, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  return run1([&](id<MTLComputeCommandEncoder> e) {
    encodeKvAppend(e, p.kvAppend, buffer(k), buffer(v), buffer(Kc), buffer(Vc), buffer(pos), B, T, hkv);
  });
}
static pmpp::Timing opAttention(Attn a, Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int B, int hq, int hkv, int T,
                                float scale, std::string* err) {
  err->clear();
  id<MTLComputePipelineState> p = attentionPso(a, hq / hkv, err);
  if (!p) return {0, -1};
  EncodeAttentionFn f = attentionEncoder(a);
  return run1([&](id<MTLComputeCommandEncoder> e) {
    f(e, p, buffer(q), buffer(Kc), buffer(Vc), buffer(len), buffer(out), B, hq, hkv, T, scale);
  });
}
static pmpp::Timing opResidual(Buf* X, Buf* R, int n, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  return run1([&](id<MTLComputeCommandEncoder> e) { encodeAddResidual(e, p.addResidual, buffer(X), buffer(R), n); });
}
static pmpp::Timing opSwiGLU(Buf* G, Buf* U, Buf* Y, int n, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  return run1([&](id<MTLComputeCommandEncoder> e) { encodeSwiGLU(e, p.swiglu, buffer(G), buffer(U), buffer(Y), n); });
}
static pmpp::Timing opArgmax(Buf* x, Buf* idx, int rows, int cols, std::string* err) {
  LlamaPipelines& p = bases(err);
  if (!err->empty()) return {0, -1};
  const NSUInteger w = p.argmax.threadExecutionWidth;
  if (p.argmax.maxTotalThreadsPerThreadgroup < 256 || w == 0 || 256 % w != 0 || 256 / w > w) {
    *err = "lab 10.3's argmax_rows cannot run 256 threads in 1 to SIMD-width whole SIMD-groups here";
    return {0, -1};
  }
  bool ok = true;
  pmpp::Timing t = run1([&](id<MTLComputeCommandEncoder> e) { ok = encodeArgmaxRows(e, p.argmax, buffer(x), buffer(idx), rows, cols); });
  if (!ok) *err = "lab 10.3's encodeArgmaxRows refused to dispatch";
  return t;
}

#if defined(PMPP_HAVE_M5)
// Lab X1.3's pipelines, as its driver builds them: its kernels from its solution's
// kernel.metal, the earlier labs' from their solution kernel.metal files.
static SamplerPipelines samplerPipes(std::string* err) {
  static SamplerPipelines p;
  static std::string problem;
  static bool built = false;
  if (!built) {
    built = true;
    const std::string labs = PMPP_LABS_DIR, mine = PMPP_M5_METAL;
    auto Q = [](const std::string& f, const char* fn) { return pmpp::metal::pipeline(f, fn); };
    const std::string c9 = labs + "/ch09/lab3_tokencounts/solution/kernel.metal";
    const std::string c10 = labs + "/ch10/lab3_softmax_argmax/solution/kernel.metal";
    const std::string c11 = labs + "/ch11/lab6_topp/solution/kernel.metal";
    const std::string c14 = labs + "/ch14/lab3_sortrows/solution/kernel.metal";
    const std::string g11 = labs + "/x1/shared-g11/";
    p.copyRows = Q(mine, "copy_rows");
    p.applyRepetition = Q(mine, "apply_repetition");
    p.drawRows = Q(mine, "draw_rows");
    p.philoxKat = Q(mine, "philox_kat");
    p.applyPenalties = Q(c9, "apply_penalties");
    p.appendTokens = Q(c9, "append_tokens");
    p.argmax = Q(c10, "argmax_rows");
    p.softmax = Q(c10, "softmax_rows");
    p.topkThreshold = Q(g11 + "topk_g11.metal", "topk_threshold");
    p.topkGather = Q(g11 + "topk_g11.metal", "topk_gather");
    p.descKeys = Q(c14, "desc_keys");
    p.sortPass = Q(c14, "sort_rows_pass");
    p.sortGather = Q(c14, "gather_rows");
    p.topP = Q(c11, "top_p_mask");
    p.minP = Q(g11 + "minp_g11.metal", "minp_combine");
    if (!p.copyRows || !p.applyRepetition || !p.drawRows || !p.philoxKat || !p.applyPenalties || !p.appendTokens ||
        !p.argmax || !p.softmax || !p.topkThreshold || !p.topkGather || !p.descKeys || !p.sortPass ||
        !p.sortGather || !p.topP || !p.minP)
      problem = "lab X1.3's Metal kernels did not all build: " + pmpp::take_error();
  }
  *err = problem;
  return p;
}
struct SamplerCtx {
  SamplerSettings set;
  SamplerPipelines pipes;
  SampleWorkspace ws;
  Buf *counts = nullptr, *seeds = nullptr, *streams = nullptr;
};
static bool encodeSampleCtx(id<MTLComputeCommandEncoder> enc, void* ctx, id<MTLBuffer> logits, id<MTLBuffer> tokens,
                            id<MTLBuffer> rngStep, uint32_t batch, uint32_t vocab) {
  SamplerCtx* c = (SamplerCtx*)ctx;
  return encodeSampleRows(enc, c->pipes, c->ws, logits, buffer(c->counts), buffer(c->seeds), buffer(c->streams), rngStep,
                          tokens, nil, batch, vocab, c->set.topK, c->set.topP, c->set.minP, c->set.temperature, 1.0f,
                          0.0f, 0.0f);
}
#endif

struct Runner {
  LlamaModel m{};
  std::vector<LlamaLayer> layers;
  std::string problem;
  Gemm gemm;
  Attn attn;
  Runner(DevModel& dm, Gemm g, Attn a, int ropeMode) : gemm(g), attn(a) {
    m.cfg = dm.c;
    m.E = buffer(dm.E);
    m.rmsFinal = buffer(dm.rmsFinal);
    for (auto& w : dm.L) {
      LlamaLayer l;
      l.rmsAtt = buffer(w[W_RMS_ATT]);
      l.wq = buffer(w[W_Q]);
      l.wk = buffer(w[W_K]);
      l.wv = buffer(w[W_V]);
      l.wo = buffer(w[W_O]);
      l.rmsFfn = buffer(w[W_RMS_FFN]);
      l.w1 = buffer(w[W_1]);
      l.w2 = buffer(w[W_2]);
      l.w3 = buffer(w[W_3]);
      layers.push_back(l);
    }
    m.layer = layers.data();
    m.ropeMode = (uint32_t)ropeMode;
    m.eps = 1e-5f;
    m.encodeLinear = linearEncoder(g);
    m.encodeAttention = attentionEncoder(a);
    std::string e;
    m.pipelines = bases(&e);
    if (e.empty()) m.pipelines.linear = linearPso(g, &e);
    if (e.empty()) m.pipelines.attention = attentionPso(a, dm.c.n_heads / dm.c.n_kv_heads, &e);
    if (e.empty()) m.pipelines.advance = advancePso(&e);
    problem = e;
  }
  Runner(const Runner&) = delete;
  void* sampler = nullptr;   // a SamplerCtx*, or null for greedy
  LlamaState view(DevState& s) {
    LlamaState v{};
    v.batch = (uint32_t)s.batch;
    v.tokens = buffer(s.tokens);
    v.pos = buffer(s.pos);
    v.len = buffer(s.len);
    v.prompt = buffer(s.prompt);
    v.promptLen = (uint32_t)s.promptLen;
    v.out = buffer(s.out);
    v.rngStep = buffer(s.rngStep);
    v.x = buffer(s.x);
    v.xb = buffer(s.xb);
    v.xb2 = buffer(s.xb2);
    v.q = buffer(s.q);
    v.k = buffer(s.k);
    v.v = buffer(s.v);
    v.att = buffer(s.att);
    v.hb = buffer(s.hb);
    v.hb2 = buffer(s.hb2);
    v.logits = buffer(s.logits);
    NSMutableArray<id<MTLBuffer>>* K = [NSMutableArray array];
    NSMutableArray<id<MTLBuffer>>* V = [NSMutableArray array];
    for (size_t l = 0; l < s.Kc.size(); ++l) {
      [K addObject:buffer(s.Kc[l])];
      [V addObject:buffer(s.Vc[l])];
    }
    v.Kc = K;
    v.Vc = V;
    v.cosT = buffer(s.cosT);
    v.sinT = buffer(s.sinT);
#if defined(PMPP_HAVE_M5)
    if (sampler) {
      v.encodeSample = encodeSampleCtx;
      v.sampleCtx = sampler;
    }
#endif
    return v;
  }
  // One step: one command buffer, timed by the GPU.
  pmpp::Timing step(DevState& s, bool* ok) {
    LlamaState v = view(s);
    *ok = true;
    return pmpp::run([&](void* pass) { *ok = encodeLlamaStep(encoderOf(pass), m, v); });
  }
  // `steps` steps with no host work in between: one command buffer per step, all committed
  // to one queue before the first wait (a queue "maintains an ordered list of command
  // buffers", infra-facts C12; buffers are hazard-tracked, ch02-facts C9).
  std::string generate(DevState& s, int steps) {
    static id<MTLCommandQueue> queue = [pmpp::metal::device() newCommandQueue];
    LlamaState v = view(s);
    NSMutableArray<id<MTLCommandBuffer>>* cbs = [NSMutableArray array];
    for (int i = 0; i < steps; ++i) {
      id<MTLCommandBuffer> cb = [queue commandBuffer];
      id<MTLComputeCommandEncoder> enc = [cb computeCommandEncoder];
      const bool ok = encodeLlamaStep(enc, m, v);
      [enc endEncoding];
      [cb commit];
      [cbs addObject:cb];
      if (!ok) {
        [cb waitUntilCompleted];
        return fmt("encodeLlamaStep returned false at step %.0f: a piece refused to dispatch", i);
      }
    }
    for (id<MTLCommandBuffer> cb in cbs) {
      [cb waitUntilCompleted];
      if (cb.status == MTLCommandBufferStatusError)
        return std::string("a command buffer failed: ") +
               (cb.error ? cb.error.localizedDescription.UTF8String : "unknown error");
    }
    return "";
  }
  std::string stepRefused() const { return "encodeLlamaStep returned false: a piece refused to dispatch"; }
};

static pmpp::Timing opAdvance(DevState& s, std::string* err) {
  id<MTLComputePipelineState> adv = advancePso(err);
  if (!adv) return {0, -1};
  LlamaState v{};
  v.batch = (uint32_t)s.batch;
  v.tokens = buffer(s.tokens);
  v.pos = buffer(s.pos);
  v.len = buffer(s.len);
  v.prompt = buffer(s.prompt);
  v.promptLen = (uint32_t)s.promptLen;
  v.out = buffer(s.out);
  v.rngStep = buffer(s.rngStep);
  return run1([&](id<MTLComputeCommandEncoder> e) { encodeAdvancePositions(e, adv, v); });
}

static double deviceMemory() { return pmpp::device_prop("recommendedMaxWorkingSetSize", 0); }
static const char* stepUnit() { return "one command buffer per step, GPUEndTime - GPUStartTime"; }

#else   // CUDA, HIP, emulators
// The provided pieces llama.h does not declare (renamed by pieces.cmake).
void launchLinearM2(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 6.4's
#if defined(PMPP_HAVE_M3)
void launchLinearM3(const float* X, const float* W, float* Y, int batch, int in, int out);   // lab 15.6's
#endif
#if defined(PMPP_HAVE_GEMV)
void launchGemvGather(const float* X, const float* W, float* Y, int batch, int in, int out); // lab 22.1's
#endif
void launchDecodeAttention(const float* q, const float* Kc, const float* Vc, const int* len, float* out, int batch,
                           int h_q, int h_kv, int max_seq, float scale);                    // lab 20.4's
#if defined(PMPP_HAVE_GQA)
void launchDecodeAttentionGrouped(const float* q, const float* Kc, const float* Vc, const int* len, float* out,
                                  int batch, int h_q, int h_kv, int max_seq, float scale);  // lab 20.5's
#endif
#if defined(PMPP_HAVE_M5)
size_t sampleWorkspaceBytes(int rows, int vocab, int topK);                                 // lab X1.3's
void launchSampleRows(const float* logits, const unsigned int* counts, const unsigned long long* seeds,
                      const unsigned long long* streams, const unsigned long long* steps, int* tokens, float* uOut,
                      void* workspace, int rows, int vocab, int topK, float topP, float minP, float temperature,
                      float repeatPenalty, float frequencyPenalty, float presencePenalty);
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
static AttentionFn attentionFor(Attn a) {
#if defined(PMPP_HAVE_GQA)
  if (a == GROUPED) return launchDecodeAttentionGrouped;
#endif
  return a == DECODE ? launchDecodeAttention : nullptr;
}
static bool gemmPresent(Gemm g) { return linearFor(g) != nullptr; }
static bool attnPresent(Attn a) { return attentionFor(a) != nullptr; }
static bool samplerPresent() {
#if defined(PMPP_HAVE_M5)
  return true;
#else
  return false;
#endif
}

static float* f32(Buf* b) { return (float*)pmpp::raw(b); }
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

static pmpp::Timing opEmbed(Buf* E, Buf* tok, Buf* H, int B, int d, std::string* err) {
  err->clear();
  return exec([&] { launchEmbed(f32(E), i32(tok), f32(H), B, d); });
}
static pmpp::Timing opRmsNorm(Buf* X, Buf* w, Buf* Y, int rows, int d, float eps, std::string* err) {
  err->clear();
  return exec([&] { launchRmsNorm(f32(X), f32(w), f32(Y), rows, d, eps); });
}
static pmpp::Timing opLinear(Gemm g, Buf* X, Buf* W, Buf* Y, int B, int in, int out, std::string* err) {
  err->clear();
  LinearFn f = linearFor(g);
  if (!f) {
    *err = std::string("no ") + gemmName(g) + " on this build";
    return {0, -1};
  }
  return exec([&] { f(f32(X), f32(W), f32(Y), B, in, out); });
}
static pmpp::Timing opRope(Buf* X, Buf* cs, Buf* sn, Buf* pos, int B, int heads, int hd, int mode, std::string* err) {
  err->clear();
  return exec([&] { launchRope(f32(X), i32(pos), f32(cs), f32(sn), B, heads, hd, mode); });
}
static pmpp::Timing opKvAppend(Buf* k, Buf* v, Buf* Kc, Buf* Vc, Buf* pos, int B, int T, int hkv, std::string* err) {
  err->clear();
  return exec([&] { launchKvAppend(f32(k), f32(v), f32(Kc), f32(Vc), i32(pos), B, T, hkv); });
}
static pmpp::Timing opAttention(Attn a, Buf* q, Buf* Kc, Buf* Vc, Buf* len, Buf* out, int B, int hq, int hkv, int T,
                                float scale, std::string* err) {
  err->clear();
  AttentionFn f = attentionFor(a);
  if (!f) {
    *err = "lab 20.5's solution was not found when the lab was configured";
    return {0, -1};
  }
  return exec([&] { f(f32(q), f32(Kc), f32(Vc), i32(len), f32(out), B, hq, hkv, T, scale); });
}
static pmpp::Timing opResidual(Buf* X, Buf* R, int n, std::string* err) {
  err->clear();
  return exec([&] { launchAddResidual(f32(X), f32(R), n); });
}
static pmpp::Timing opSwiGLU(Buf* G, Buf* U, Buf* Y, int n, std::string* err) {
  err->clear();
  return exec([&] { launchSwiGLU(f32(G), f32(U), f32(Y), n); });
}
static pmpp::Timing opArgmax(Buf* x, Buf* idx, int rows, int cols, std::string* err) {
  err->clear();
  return exec([&] { launchArgmaxRows(f32(x), i32(idx), rows, cols); });
}
static pmpp::Timing opAdvance(DevState& s, std::string* err) {
  err->clear();
  return exec([&] {
    launchAdvancePositions(i32(s.tokens), i32(s.pos), i32(s.len), i32(s.prompt), s.promptLen, i32(s.out),
                           (unsigned long long*)pmpp::raw(s.rngStep), s.batch);
  });
}

#if defined(PMPP_HAVE_M5)
struct SamplerCtx {
  SamplerSettings set;
  Buf *counts = nullptr, *seeds = nullptr, *streams = nullptr, *ws = nullptr;
};
static void sampleCtx(void* ctx, const float* logits, int* tokens, const unsigned long long* rngStep, int batch,
                      int vocab) {
  SamplerCtx* c = (SamplerCtx*)ctx;
  launchSampleRows(logits, (const unsigned int*)pmpp::raw(c->counts), (const unsigned long long*)pmpp::raw(c->seeds),
                   (const unsigned long long*)pmpp::raw(c->streams), rngStep, tokens, nullptr, pmpp::raw(c->ws), batch,
                   vocab, c->set.topK, c->set.topP, c->set.minP, c->set.temperature, 1.0f, 0.0f, 0.0f);
}
#endif

struct Runner {
  LlamaModel m{};
  std::vector<LlamaLayer> layers;
  std::vector<float*> Kc, Vc;
  std::string problem;
  Gemm gemm;
  Attn attn;
  void* sampler = nullptr;   // a SamplerCtx*, or null for greedy
  Runner(DevModel& dm, Gemm g, Attn a, int ropeMode) : gemm(g), attn(a) {
    m.cfg = dm.c;
    m.E = f32(dm.E);
    m.rmsFinal = f32(dm.rmsFinal);
    for (auto& w : dm.L) {
      LlamaLayer l;
      l.rmsAtt = f32(w[W_RMS_ATT]);
      l.wq = f32(w[W_Q]);
      l.wk = f32(w[W_K]);
      l.wv = f32(w[W_V]);
      l.wo = f32(w[W_O]);
      l.rmsFfn = f32(w[W_RMS_FFN]);
      l.w1 = f32(w[W_1]);
      l.w2 = f32(w[W_2]);
      l.w3 = f32(w[W_3]);
      layers.push_back(l);
    }
    m.layer = layers.data();
    m.linear = linearFor(g);
    m.attention = attentionFor(a);
    m.ropeMode = ropeMode;
    m.eps = 1e-5f;
    if (!m.linear) problem = std::string("no ") + gemmName(g) + " on this build";
    else if (!m.attention) problem = "lab 20.5's solution was not found when the lab was configured";
  }
  Runner(const Runner&) = delete;
  LlamaState view(DevState& s) {
    Kc.clear();
    Vc.clear();
    for (size_t l = 0; l < s.Kc.size(); ++l) {
      Kc.push_back(f32(s.Kc[l]));
      Vc.push_back(f32(s.Vc[l]));
    }
    LlamaState v{};
    v.batch = s.batch;
    v.tokens = i32(s.tokens);
    v.pos = i32(s.pos);
    v.len = i32(s.len);
    v.prompt = i32(s.prompt);
    v.promptLen = s.promptLen;
    v.out = i32(s.out);
    v.rngStep = (unsigned long long*)pmpp::raw(s.rngStep);
    v.x = f32(s.x);
    v.xb = f32(s.xb);
    v.xb2 = f32(s.xb2);
    v.q = f32(s.q);
    v.k = f32(s.k);
    v.v = f32(s.v);
    v.att = f32(s.att);
    v.hb = f32(s.hb);
    v.hb2 = f32(s.hb2);
    v.logits = f32(s.logits);
    v.Kc = Kc.data();
    v.Vc = Vc.data();
    v.cosT = f32(s.cosT);
    v.sinT = f32(s.sinT);
#if defined(PMPP_HAVE_M5)
    if (sampler) {
      v.sample = sampleCtx;
      v.sampleCtx = sampler;
    }
#endif
    return v;
  }
  pmpp::Timing step(DevState& s, bool* ok) {
    *ok = true;
    LlamaState v = view(s);
    return exec([&] { launchLlamaStep(m, v); });
  }
  // `steps` steps queued back to back on the default stream: one event pair, one wait.
  std::string generate(DevState& s, int steps) {
    LlamaState v = view(s);
    exec([&] {
      for (int i = 0; i < steps; ++i) launchLlamaStep(m, v);
    });
    return "";
  }
};

static double deviceMemory() { return pmpp::device_prop("totalGlobalMem", 0); }
static const char* stepUnit() { return "one pair of events per step"; }
#endif

// ---------------------------------------------------------------- the sampler's state, both backends
#if defined(PMPP_HAVE_M5)
struct SamplerHolder {
  SamplerCtx ctx;
  std::string problem;
  SamplerHolder(int batch, int vocab, const SamplerSettings& set) {
    ctx.set = set;
    pmpp::take_error();
    ctx.counts = pmpp::alloc((size_t)batch * vocab * 4);
    ctx.seeds = pmpp::alloc((size_t)batch * 8);
    ctx.streams = pmpp::alloc((size_t)batch * 8);
    std::vector<uint32_t> zeros((size_t)batch * vocab, 0);
    pmpp::upload(ctx.counts, zeros.data(), zeros.size() * 4);   // no penalties: the counts are never read
    std::vector<unsigned long long> seeds(batch, 0x5EEDull), streams(batch);
    for (int b = 0; b < batch; ++b) streams[b] = (unsigned long long)b;
    pmpp::upload(ctx.seeds, seeds.data(), seeds.size() * 8);
    pmpp::upload(ctx.streams, streams.data(), streams.size() * 8);
#if defined(PMPP_BACKEND_METAL)
    ctx.pipes = samplerPipes(&problem);
    ctx.ws = makeSampleWorkspace(pmpp::metal::device(), (uint32_t)batch, (uint32_t)vocab, set.topK);
#else
    ctx.ws = pmpp::alloc(std::max<size_t>(16, sampleWorkspaceBytes(batch, vocab, set.topK)));
#endif
    std::string e = pmpp::take_error();
    if (problem.empty() && !e.empty()) problem = "could not allocate the sampler's buffers: " + e;
  }
  SamplerHolder(const SamplerHolder&) = delete;
  ~SamplerHolder() {
    pmpp::release(ctx.counts);
    pmpp::release(ctx.seeds);
    pmpp::release(ctx.streams);
#if !defined(PMPP_BACKEND_METAL)
    pmpp::release(ctx.ws);
#endif
  }
};
#endif

// ---------------------------------------------------------------- the pieces alone
// Each provided piece once, against a CPU reference, before your step is tested: a failure
// here is the course's, not yours.
struct Close {
  size_t bad = 0, first = 0;
  double got = 0, want = 0, maxErr = 0;
  void see(size_t i, double g, double w, double mag, double rtol) {
    const double e = std::fabs(g - w);
    if (!(e <= rtol * (1.0 + mag))) {
      if (bad++ == 0) {
        first = i;
        got = g;
        want = w;
      }
    }
    if (e > maxErr || std::isnan(e)) maxErr = e;
  }
  std::string why(const std::string& what) const {
    if (!bad) return "";
    return what + fmt(": %.0f wrong elements, first at %.0f: got %.7g, expected %.7g", (double)bad, (double)first, got,
                      want);
  }
};
static std::vector<float> rnd(size_t n, uint32_t seed, float scale = 1.0f) {
  std::vector<float> v = pmpp::random_floats(n, seed);
  for (float& x : v) x *= scale;
  return v;
}
// A device copy of `v` with a guard band (an output) or at its exact size (an input).
static Buf* dev(const std::vector<float>& v, bool guard) {
  Buf* b = allocWords(v.size(), guard);
  if (guard) uploadGuarded(b, v);
  else pmpp::upload(b, v.data(), v.size() * 4);
  return b;
}
static Buf* devInt(const std::vector<int>& v, bool guard) {
  Buf* b = allocWords(v.size(), guard);
  if (guard) uploadGuarded(b, v);
  else pmpp::upload(b, v.data(), v.size() * 4);
  return b;
}
struct Bufs {
  std::vector<Buf*> list;
  Buf* operator()(Buf* b) {
    list.push_back(b);
    return b;
  }
  ~Bufs() {
    for (Buf* b : list) pmpp::release(b);
  }
};
static std::string after(const std::string& err) {
  if (!err.empty()) return err;
  return pmpp::take_error();
}
static std::vector<float> nanVec(size_t n) {
  float q;
  std::memcpy(&q, &kQuietNaN, 4);
  return std::vector<float>(n, q);
}

static std::string pieceEmbed(const HostModel& h) {
  Bufs B;
  const int d = h.c.dim, V = h.c.vocab_size;
  const std::vector<int> tok = {0, V - 1, 7, 7};
  const int n = (int)tok.size();
  Buf* E = B(dev(h.E, false));
  Buf* t = B(devInt(tok, false));
  Buf* H = B(dev(nanVec((size_t)n * d), true));
  std::string err;
  opEmbed(E, t, H, n, d, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(H, (size_t)n * d);
  for (int r = 0; r < n; ++r)
    for (int j = 0; j < d; ++j)
      if (std::memcmp(&got[(size_t)r * d + j], &h.E[(size_t)tok[r] * d + j], 4) != 0)
        return fmt("row %.0f is not row %.0f of E (first difference at feature %.0f)", r, tok[r], j);
  return guardIntact(got, (size_t)n * d) ? "" : "lab 22.2's embed wrote past the end of its output";
}
static std::string pieceRmsNorm(const HostModel& h) {
  Bufs B;
  const int d = h.c.dim, rows = 3;
  const std::vector<float> x = rnd((size_t)rows * d, 31, 2.0f);
  const std::vector<float>& w = h.L[0].rmsAtt;
  Buf* X = B(dev(x, false));
  Buf* W = B(dev(w, false));
  Buf* Y = B(dev(nanVec(x.size()), true));
  std::string err;
  opRmsNorm(X, W, Y, rows, d, 1e-5f, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(Y, x.size());
  Close c;
  for (int r = 0; r < rows; ++r) {
    double ss = 0;
    for (int j = 0; j < d; ++j) ss += (double)x[r * d + j] * x[r * d + j];
    const double inv = 1.0 / std::sqrt(ss / d + kEps);
    for (int j = 0; j < d; ++j) {
      const double want = w[j] * (inv * x[r * d + j]);
      c.see((size_t)r * d + j, got[(size_t)r * d + j], want, std::fabs(want), kPieceRtol);
    }
  }
  if (!guardIntact(got, x.size())) return "lab X3.1's RMSNorm wrote past the end of its output";
  return c.why("lab X3.1's RMSNorm");
}
static std::string pieceLinear(const HostModel& h, Gemm g, int batch) {
  Bufs B;
  const int in = h.c.dim, out = dqOf(h.c);
  const std::vector<float> x = rnd((size_t)batch * in, 41);
  const std::vector<float>& w = h.L[0].wq;
  Buf* X = B(dev(x, false));
  Buf* W = B(dev(w, false));
  Buf* Y = B(dev(nanVec((size_t)batch * out), true));
  std::string err;
  opLinear(g, X, W, Y, batch, in, out, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(Y, (size_t)batch * out);
  Close c;
  for (int r = 0; r < batch; ++r)
    for (int o = 0; o < out; ++o) {
      double acc = 0, mag = 0;
      for (int i = 0; i < in; ++i) {
        acc += (double)x[r * in + i] * w[(size_t)o * in + i];
        mag += std::fabs((double)x[r * in + i] * w[(size_t)o * in + i]);
      }
      c.see((size_t)r * out + o, got[(size_t)r * out + o], acc, mag, kPieceRtol);
    }
  if (!guardIntact(got, (size_t)batch * out)) return std::string(gemmSource(g)) + " wrote past the end of its output";
  return c.why(gemmSource(g));
}
static std::string pieceRope(const DevState& s0, const HostModel& h, int mode) {
  Bufs B;
  const int heads = h.c.n_heads, hd = LLAMA_HEAD_DIM, half = hd / 2, T = h.c.seq_len;
  const std::vector<int> pos = {3, T - 1};
  const int batch = (int)pos.size();
  const std::vector<float> x = rnd((size_t)batch * heads * hd, 51, 1.5f);
  Buf* X = B(dev(x, true));
  Buf* P = B(devInt(pos, false));
  std::string err;
  opRope(X, s0.cosT, s0.sinT, P, batch, heads, hd, mode, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(X, x.size());
  Close c;
  for (int b = 0; b < batch; ++b)
    for (int hh = 0; hh < heads; ++hh)
      for (int i = 0; i < half; ++i) {
        const int i0 = mode == LLAMA_ROPE_CONSECUTIVE ? 2 * i : i, i1 = mode == LLAMA_ROPE_CONSECUTIVE ? 2 * i + 1 : i + half;
        const size_t o = ((size_t)b * heads + hh) * hd;
        const double a = pos[b] * ropeFreq(i), cs = std::cos(a), sn = std::sin(a), x0 = x[o + i0], x1 = x[o + i1];
        const double mag = std::fabs(x0) + std::fabs(x1);
        c.see(o + i0, got[o + i0], x0 * cs - x1 * sn, mag, kPieceRtol);
        c.see(o + i1, got[o + i1], x0 * sn + x1 * cs, mag, kPieceRtol);
      }
  if (!guardIntact(got, x.size())) return "lab X3.1's RoPE wrote past the end of its input";
  return c.why(std::string("lab X3.1's RoPE, mode ") + std::to_string(mode));
}
// Lab 20.4's append alone, then the attention alone over a cache whose unread rows are NaN.
static std::string pieceKvAppend(const HostModel& h) {
  Bufs B;
  const int T = h.c.seq_len, dkv = dkvOf(h.c), hkv = h.c.n_kv_heads;
  const std::vector<int> pos = {0, T - 1};
  const int batch = (int)pos.size();
  const std::vector<float> k = rnd((size_t)batch * dkv, 61), v = rnd((size_t)batch * dkv, 62);
  Buf* Kn = B(dev(k, false));
  Buf* Vn = B(dev(v, false));
  Buf* P = B(devInt(pos, false));
  Buf* Kc = B(dev(nanVec((size_t)batch * T * dkv), true));
  Buf* Vc = B(dev(nanVec((size_t)batch * T * dkv), true));
  std::string err;
  opKvAppend(Kn, Vn, Kc, Vc, P, batch, T, hkv, &err);
  if (!(err = after(err)).empty()) return err;
  for (int which = 0; which < 2; ++which) {
    std::vector<float> got = fetch<float>(which ? Vc : Kc, (size_t)batch * T * dkv);
    const std::vector<float>& src = which ? v : k;
    for (int b = 0; b < batch; ++b)
      for (int t = 0; t < T; ++t)
        for (int i = 0; i < dkv; ++i) {
          const float g = got[((size_t)b * T + t) * dkv + i];
          if (t == pos[b] ? std::memcmp(&g, &src[(size_t)b * dkv + i], 4) != 0 : !std::isnan(g))
            return fmt("lab 20.4's append: sequence %.0f, row %.0f, element %.0f is wrong", b, t, i);
        }
    if (!guardIntact(got, (size_t)batch * T * dkv)) return "lab 20.4's append wrote past the end of a cache";
  }
  return "";
}
static std::string pieceAttention(const HostModel& h, Attn a) {
  Bufs B;
  const int T = h.c.seq_len, dkv = dkvOf(h.c), dq = dqOf(h.c), hq = h.c.n_heads, hkv = h.c.n_kv_heads, hd = LLAMA_HEAD_DIM;
  const std::vector<int> len = {3, T};
  const int batch = (int)len.size();
  const std::vector<float> q = rnd((size_t)batch * dq, 71, 0.5f);
  std::vector<float> K = nanVec((size_t)batch * T * dkv), V = K;
  const std::vector<float> kr = rnd(K.size(), 72), vr = rnd(K.size(), 73);
  for (int b = 0; b < batch; ++b)
    for (int t = 0; t < len[b]; ++t)
      for (int i = 0; i < dkv; ++i) {
        const size_t o = ((size_t)b * T + t) * dkv + i;
        K[o] = kr[o];
        V[o] = vr[o];
      }
  Buf* Q = B(dev(q, false));
  Buf* Kc = B(dev(K, false));
  Buf* Vc = B(dev(V, false));
  Buf* L = B(devInt(len, false));
  Buf* O = B(dev(nanVec((size_t)batch * dq), true));
  const float scale = 1.0f / std::sqrt((float)hd);
  std::string err;
  opAttention(a, Q, Kc, Vc, L, O, batch, hq, hkv, T, scale, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(O, (size_t)batch * dq);
  Close c;
  for (int b = 0; b < batch; ++b)
    for (int hh = 0; hh < hq; ++hh) {
      const int kvh = hh / (hq / hkv);
      std::vector<double> s(len[b]);
      double mx = -1e300, z = 0;
      for (int t = 0; t < len[b]; ++t) {
        double dot = 0;
        for (int i = 0; i < hd; ++i) dot += (double)q[(b * hq + hh) * hd + i] * K[((size_t)b * T + t) * dkv + kvh * hd + i];
        s[t] = dot * scale;
        mx = std::max(mx, s[t]);
      }
      for (int t = 0; t < len[b]; ++t) z += (s[t] = std::exp(s[t] - mx));
      for (int i = 0; i < hd; ++i) {
        double o = 0, mag = 0;
        for (int t = 0; t < len[b]; ++t) {
          const double vv = V[((size_t)b * T + t) * dkv + kvh * hd + i];
          o += s[t] / z * vv;
          mag += s[t] / z * std::fabs(vv);
        }
        c.see((size_t)(b * hq + hh) * hd + i, got[(size_t)(b * hq + hh) * hd + i], o, mag, kPieceRtol);
      }
    }
  if (!guardIntact(got, (size_t)batch * dq)) return std::string(attnName(a)) + " wrote past the end of its output";
  return c.why(attnName(a));
}
static std::string pieceResidual(const HostModel& h) {
  Bufs B;
  const int n = 2 * h.c.dim + 3;
  const std::vector<float> x = rnd(n, 81), r = rnd(n, 82);
  Buf* X = B(dev(x, true));
  Buf* R = B(dev(r, false));
  std::string err;
  opResidual(X, R, n, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(X, n);
  for (int i = 0; i < n; ++i) {
    const float want = x[i] + r[i];
    if (std::memcmp(&got[i], &want, 4) != 0) return fmt("lab 2.3's add_residual: element %.0f is wrong", i);
  }
  return guardIntact(got, n) ? "" : "lab 2.3's add_residual wrote past the end of X";
}
static std::string pieceSwiGLU(const HostModel& h) {
  Bufs B;
  const int n = 2 * h.c.hidden_dim + 5;
  const std::vector<float> g = rnd(n, 91, 4.0f), u = rnd(n, 92, 2.0f);
  Buf* G = B(dev(g, false));
  Buf* U = B(dev(u, false));
  Buf* Y = B(dev(nanVec(n), true));
  std::string err;
  opSwiGLU(G, U, Y, n, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<float> got = fetch<float>(Y, n);
  Close c;
  for (int i = 0; i < n; ++i) {
    const double want = g[i] / (1.0 + std::exp(-(double)g[i])) * u[i];
    c.see(i, got[i], want, std::fabs(want), kPieceRtol);
  }
  if (!guardIntact(got, n)) return "lab X3.1's SwiGLU wrote past the end of its output";
  return c.why("lab X3.1's SwiGLU");
}
static std::string pieceArgmax(const HostModel& h) {
  Bufs B;
  const int V = h.c.vocab_size, rows = 3;
  std::vector<float> x = rnd((size_t)rows * V, 101);
  x[0 * V + 9] = 2.0f;           // row 0: a tie between columns 9 and V - 2
  x[0 * V + V - 2] = 2.0f;
  x[1 * V + V - 1] = 2.0f;       // row 1: the last column
  x[2 * V + 0] = 2.0f;           // row 2: the first column
  Buf* X = B(dev(x, false));
  Buf* I = B(devInt(std::vector<int>(rows, kNoToken), true));
  std::string err;
  opArgmax(X, I, rows, V, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<int> got = fetch<int>(I, rows);
  const int want[3] = {9, V - 1, 0};
  for (int r = 0; r < rows; ++r)
    if (got[r] != want[r]) return fmt("lab 10.3's argmax: row %.0f gave %.0f, expected %.0f", r, got[r], want[r]);
  return guardIntact(got, rows) ? "" : "lab 10.3's argmax wrote past the end of idx";
}

// Your advance alone: three sequences at positions 0, 2 and 5 of a 4-token prompt.
static std::string checkAdvance(const LlamaConfig& c) {
  const int batch = 3, P = 4, T = c.seq_len;
  std::vector<int> prompt(P * batch);
  for (int i = 0; i < P * batch; ++i) prompt[i] = 100 + i;
  DevState s(c, batch, prompt, P, true);
  if (!s.ok) return "could not allocate the state";
  const std::vector<int> pos = {0, 2, 5}, tok = {7, 8, 9};
  uploadGuarded(s.tokens, tok);
  uploadGuarded(s.pos, pos);
  uploadGuarded(s.len, std::vector<int>{1, 3, 6});
  uploadGuarded(s.rngStep, std::vector<uint32_t>{0, 0, 2, 0, 5, 0});
  std::string err;
  opAdvance(s, &err);
  if (!(err = after(err)).empty()) return err;
  std::vector<int> t = fetch<int>(s.tokens, batch), p = fetch<int>(s.pos, batch), l = fetch<int>(s.len, batch);
  std::vector<int> o = fetch<int>(s.out, (size_t)T * batch);
  std::vector<uint32_t> r = fetch<uint32_t>(s.rngStep, 2 * batch);
  for (int b = 0; b < batch; ++b) {
    const int wantTok = pos[b] + 1 < P ? prompt[(pos[b] + 1) * batch + b] : tok[b];
    if (o[(size_t)pos[b] * batch + b] != tok[b])
      return fmt("sequence %.0f at position %.0f: out[%.0f] is %.0f, expected this step's token",
                 b, pos[b], pos[b] * batch + b, o[(size_t)pos[b] * batch + b]);
    if (t[b] != wantTok)
      return fmt("sequence %.0f at position %.0f of a %.0f-token prompt: next token %.0f, expected ", b, pos[b], P, t[b]) +
             (pos[b] + 1 < P ? "the prompt's token at the next position" : "this step's token, unchanged");
    if (p[b] != pos[b] + 1 || l[b] != pos[b] + 2)
      return fmt("sequence %.0f: pos %.0f, len %.0f after the advance; expected pos + 1 and pos + 2", b, p[b], l[b]);
    if (r[2 * b] != (uint32_t)pos[b] + 1 || r[2 * b + 1] != 0) return fmt("sequence %.0f: rngStep did not go up by one", b);
  }
  for (size_t i = 0; i < (size_t)T * batch; ++i) {
    const bool mine = (i == 0 * batch + 0) || (i == 2 * batch + 1) || (i == 5 * batch + 2);
    if (!mine && o[i] != kNoToken) return fmt("out[%.0f] was written; only out[pos * batch + b] should be", (double)i);
  }
  return s.guards();
}

// ---------------------------------------------------------------- the step against the reference
static std::set<std::string> gPrinted;   // the emulators run each check three times: print once
static bool firstTime(const std::string& key) { return gPrinted.insert(key).second; }

static int argmaxLowest(const float* x, int n) {
  int best = 0;
  for (int i = 1; i < n; ++i)
    if (x[i] > x[best]) best = i;
  return best;
}

// Teacher forcing: every sequence feeds the reference's token at every step (the whole
// trace is the prompt), so the logits of every step can be compared.
static std::string teacherForced(Runner& run, DevModel& dm, const HostModel& h, const std::vector<const RefTrace*>& refs,
                                 int steps, const std::string& label, bool table) {
  if (!run.problem.empty()) return run.problem;
  const int B = (int)refs.size(), V = h.c.vocab_size;
  std::vector<int> fed((size_t)steps * B);
  for (int s = 0; s < steps; ++s)
    for (int b = 0; b < B; ++b) fed[(size_t)s * B + b] = refs[b]->steps[s].fed;
  DevState st(h.c, B, fed, steps, true);
  if (!st.ok) return "could not allocate the state";
  const bool print = firstTime(label);
  if (print && table) std::printf("    %s: per step, the worst sequence\n    step  NMSE        max |err|   err/tol\n", label.c_str());
  double worstN = 0, worstE = 0, worstR = 0;
  std::string skipped;
  for (int s = 0; s < steps; ++s) {
    bool ok = true;
    run.step(st, &ok);
    std::string err = pmpp::take_error();
    if (!err.empty()) return fmt("step %.0f: ", s) + err;
    if (!ok) return fmt("step %.0f: the step refused to dispatch (a piece returned false)", s);
    std::vector<float> L = fetch<float>(st.logits, (size_t)B * V);
    std::vector<int> tok = fetch<int>(st.tokens, B), pos = fetch<int>(st.pos, B), len = fetch<int>(st.len, B);
    std::vector<int> out = fetch<int>(st.out, (size_t)h.c.seq_len * B);
    double stepN = 0, stepE = 0, stepR = 0;
    for (int b = 0; b < B; ++b) {
      const RefStep& r = refs[b]->steps[s];
      double num = 0, den = 0;
      size_t bad = 0, firstBad = 0;
      for (int v = 0; v < V; ++v) {
        const double g = L[(size_t)b * V + v], e = std::fabs(g - r.logits[v]);
        num += (g - r.logits[v]) * (g - r.logits[v]);
        den += r.logits[v] * r.logits[v];
        stepE = std::max(stepE, std::isnan(e) ? HUGE_VAL : e);
        stepR = std::max(stepR, std::isnan(e) ? HUGE_VAL : e / tolOf(r.mag[v]));
        if (!(e <= tolOf(r.mag[v])) && bad++ == 0) firstBad = v;
      }
      stepN = std::max(stepN, std::isnan(num) ? HUGE_VAL : num / den);
      if (bad)
        return fmt("step %.0f (position %.0f), sequence %.0f: %.0f logits outside the tolerance; first at token ", s, s, b,
                   (double)bad) +
               fmt("%.0f: got %.7g, expected %.7g (tolerance %.3g)", (double)firstBad, L[(size_t)b * V + firstBad],
                   r.logits[firstBad], tolOf(r.mag[firstBad]));
      const int own = argmaxLowest(&L[(size_t)b * V], V), rec = out[(size_t)s * B + b];
      if (rec != own)
        return fmt("step %.0f, sequence %.0f: out[] holds %.0f, but the argmax of this step's logits is %.0f", s, b, rec,
                   own);
      if (r.conclusive() && rec != r.argmax)
        return fmt("step %.0f, sequence %.0f: argmax %.0f, the reference's is %.0f", s, b, rec, r.argmax);
      if (!r.conclusive() && print)
        skipped += fmt(" %.0f", s);
      const int want = s + 1 < steps ? fed[(size_t)(s + 1) * B + b] : rec;
      if (tok[b] != want)
        return fmt("step %.0f, sequence %.0f: tokens[] holds %.0f afterwards, expected %.0f (the next prompt token)", s, b,
                   tok[b], want);
      if (pos[b] != s + 1 || len[b] != s + 2)
        return fmt("step %.0f, sequence %.0f: pos %.0f and len %.0f afterwards; expected step + 1 and step + 2", s, b,
                   pos[b], len[b]);
    }
    if (print && table) std::printf("    %4d  %.3e   %.3e   %.3f\n", s, stepN, stepE, stepR);
    worstN = std::max(worstN, stepN);
    worstE = std::max(worstE, stepE);
    worstR = std::max(worstR, stepR);
  }
  if (print) {
    std::printf("    %s: worst NMSE %.3e, worst max |err| %.3e, worst err/tol %.3f over %d steps\n", label.c_str(),
                worstN, worstE, worstR, steps);
    if (!skipped.empty())
      std::printf("    %s: token not checked at steps%s (reference top-two margin below twice the tolerance)\n",
                  label.c_str(), skipped.c_str());
  }
  std::string g = st.guards();
  if (!g.empty()) return g;
  g = st.cacheRows(steps);
  if (!g.empty()) return g;
  return weightsUnchanged(dm, h);
}

// Free running: the real prompt, then `steps` steps with no host work in between; the
// record is copied once at the end and compared with the reference's greedy tokens.
static std::string generateRecord(Runner& run, const LlamaConfig& c, const std::vector<const RefTrace*>& refs, int P,
                                  int steps, std::vector<int>* record, DevState** keep = nullptr) {
  if (!run.problem.empty()) return run.problem;
  const int B = (int)refs.size();
  std::vector<int> prompt((size_t)P * B);
  for (int p = 0; p < P; ++p)
    for (int b = 0; b < B; ++b) prompt[(size_t)p * B + b] = refs[b]->prompt[p];
  std::unique_ptr<DevState> st(new DevState(c, B, prompt, P, true));
  if (!st->ok) return "could not allocate the state";
  std::string err = run.generate(*st, steps);
  if (err.empty()) err = pmpp::take_error();
  if (!err.empty()) return err;
  std::vector<int> out = fetch<int>(st->out, (size_t)c.seq_len * B);   // the one copy back
  std::vector<int> pos = fetch<int>(st->pos, B), len = fetch<int>(st->len, B);
  for (int b = 0; b < B; ++b)
    if (pos[b] != steps || len[b] != steps + 1)
      return fmt("sequence %.0f: pos %.0f and len %.0f after %.0f steps", b, pos[b], len[b], steps);
  for (size_t i = (size_t)steps * B; i < (size_t)c.seq_len * B; ++i)
    if (out[i] != kNoToken) return "out[] was written beyond the rows of the steps that ran";
  record->assign(out.begin(), out.begin() + (size_t)steps * B);
  std::string g = st->guards();
  if (!g.empty()) return g;
  g = st->cacheRows(steps);
  if (!g.empty()) return g;
  if (keep) *keep = st.release();
  return "";
}

// Compares a record [steps x B] with the reference's greedy tokens. Inside the prompt a
// low-margin step is skipped alone (the next token is forced); from the first low-margin
// step of the generation on, a flip would change every later token: "inconclusive".
static std::string compareRecord(const std::vector<int>& rec, const std::vector<const RefTrace*>& refs, int P, int steps,
                                 const std::string& label, int* conclusiveSteps) {
  const int B = (int)refs.size();
  const bool print = firstTime(label);
  *conclusiveSteps = steps;
  for (int b = 0; b < B; ++b) {
    const RefTrace& r = *refs[b];
    for (int s = 0; s < steps; ++s) {
      if (!r.steps[s].conclusive()) {
        if (s >= P - 1) {
          if (print)
            std::printf("    %s, sequence %d: inconclusive from step %d on (reference margin %.3g, needed %.3g); "
                        "not checked after it\n",
                        label.c_str(), b, s, r.steps[s].gap, r.steps[s].need);
          *conclusiveSteps = std::min(*conclusiveSteps, s);
          break;
        }
        continue;
      }
      const int got = rec[(size_t)s * B + b];
      if (got != r.steps[s].argmax)
        return fmt("sequence %.0f, step %.0f: token %.0f, the reference's greedy token is %.0f", b, s, got,
                   r.steps[s].argmax) +
               fmt(" (margin %.3g, tolerance of the two %.3g)", r.steps[s].gap, r.steps[s].need);
    }
  }
  return "";
}

// A record is [steps x B] only when generateRecord succeeded; after a failed run it is empty
// or short, so check its size before indexing it (an unchecked read crashed the driver).
static std::string tokenLine(const std::vector<int>& rec, int B, int b, int from, int steps) {
  if (B <= 0 || b < 0 || b >= B || steps < 0 || rec.size() < (size_t)steps * B) return "(no record: the run did not finish)";
  std::string s;
  for (int i = from; i < steps; ++i) s += (s.empty() ? "" : " ") + std::to_string(rec[(size_t)i * B + b]);
  return s;
}

// ---------------------------------------------------------------- tests
static int tests(const pmpp::Args& a) {
  pmpp::Tests t(a);
  const bool emu = pmpp::is_emu();
  const std::string path = emu ? PMPP_X3_MODEL_EMU : PMPP_X3_MODEL_GPU;
  const std::string permPath = emu ? PMPP_X3_MODEL_EMU_PERM : PMPP_X3_MODEL_GPU_PERM;
  const std::string printed = emu ? PMPP_X3_SUM_EMU : PMPP_X3_SUM_GPU;
  const std::string permPrinted = emu ? PMPP_X3_SUM_EMU_PERM : PMPP_X3_SUM_GPU_PERM;
  const std::string recorded = emu ? PMPP_X3_EXPECT_EMU : PMPP_X3_EXPECT_GPU;
  const int seed = emu ? PMPP_X3_SEED_EMU : PMPP_X3_SEED_GPU;
  const int P = emu ? 3 : 4, G = emu ? 5 : 60, steps = P + G - 1;
  const std::vector<int> batches = emu ? std::vector<int>{1, 2} : std::vector<int>{1, 3};
  const int Bmax = batches.back();

  HostModel h, hp;
  std::string le = loadModel(path, h);
  if (le.empty()) le = loadModel(permPath, hp);
  t.check("model files: load " + std::string(emu ? "models/emu.bin" : "models/gpu.bin") + " and its --permute-qk twin",
          [&] { return le; });
  if (!le.empty()) return t.finish();
  if (steps > h.c.seq_len) {
    t.check("the test fits the model", [&] { return std::string("prompt + generated tokens exceed seq_len"); });
    return t.finish();
  }
  std::printf("test model: %s, seed %d\n  prompt %d tokens, %d generated, %d steps; batches", shape(h.c).c_str(), seed, P,
              G, steps);
  for (int B : batches) std::printf(" %d", B);
  std::printf("\n");

  t.check(fmt("model files: %.0f bytes each, as E1 computes from the header", (double)h.fileBytes), [&] {
    return h.fileBytes == e1Bytes(h.c) && hp.fileBytes == e1Bytes(hp.c) ? "" : "a file size differs from E1";
  });
  t.check("model files: the checksum of the first and last weights matches what make_model.py printed", [&] {
    if (hex64(h.checksum) != printed) return "models file " + hex64(h.checksum) + ", the script printed " + printed;
    if (hex64(hp.checksum) != permPrinted) return "the permuted file " + hex64(hp.checksum) + ", the script printed " + permPrinted;
    return std::string();
  });
  t.check("model files: the checksum matches the one recorded for the provided seed (a deterministic generator)", [&] {
    if (recorded == "unset") return std::string("no checksum recorded in provided/pieces.cmake");
    const int provided = emu ? 11 : 1;
    if (seed != provided) {
      if (firstTime("seedcheck")) std::printf("    not checked: seed %d is not the provided seed %d\n", seed, provided);
      return std::string();
    }
    return hex64(h.checksum) == recorded ? std::string()
                                         : "got " + hex64(h.checksum) + ", recorded " + recorded +
                                               ": make_model.py did not write the same bytes as when the lab was made";
  });
  t.check("model files: --permute-qk applied llama.cpp's permutation to wq and wk and changed nothing else", [&] {
    const int hd = LLAMA_HEAD_DIM, per = hd / 2, d = h.c.dim;
    for (int l = 0; l < h.c.n_layers; ++l) {
      for (int which = 0; which < 2; ++which) {
        const std::vector<float>& a0 = which ? h.L[l].wk : h.L[l].wq;
        const std::vector<float>& a1 = which ? hp.L[l].wk : hp.L[l].wq;
        const int heads = which ? h.c.n_kv_heads : h.c.n_heads;
        for (int hh = 0; hh < heads; ++hh)
          for (int half = 0; half < 2; ++half)
            for (int i = 0; i < per; ++i) {
              const size_t src = (size_t)(hh * hd + half * per + i) * d, dst = (size_t)(hh * hd + 2 * i + half) * d;
              if (std::memcmp(&a0[src], &a1[dst], (size_t)d * 4) != 0)
                return fmt("layer %.0f: row %.0f of the original is not row %.0f of the permuted ", l, hh * hd + half * per + i,
                           hh * hd + 2 * i + half) + (which ? "wk" : "wq");
            }
      }
      for (int i = 0; i < W_COUNT; ++i)
        if (i != W_Q && i != W_K && hostWeight(h.L[l], i) != hostWeight(hp.L[l], i)) return std::string("another tensor differs");
    }
    return h.E == hp.E && h.rmsFinal == hp.rmsFinal ? std::string() : std::string("E or rmsFinal differs");
  });

  DevModel dm(h);
  if (!dm.ok) {
    t.check("allocate the test model", [] { return std::string("allocation failed"); });
    return t.finish();
  }

  // The reference: greedy from each sequence's prompt, consecutive RoPE (llama2.c's pairing).
  std::vector<RefTrace> refs;
  for (int b = 0; b < Bmax; ++b) refs.push_back(refRun(h, LLAMA_ROPE_CONSECUTIVE, promptFor(b, P, h.c.vocab_size), steps));
  double minRatio = 1e300;
  int lowSteps = 0;
  for (const RefTrace& r : refs)
    for (const RefStep& s : r.steps) {
      minRatio = std::min(minRatio, s.gap / s.need);
      lowSteps += !s.conclusive();
    }
  std::printf("reference (CPU, double): sequence 0 generates %s\n",
              [&] {
                std::string s;
                for (int i = P - 1; i < steps; ++i) s += (s.empty() ? "" : " ") + std::to_string(refs[0].steps[i].argmax);
                return s;
              }().c_str());
  std::printf("  smallest top-two margin over twice the tolerance: %.1f (%d low-margin steps); logit tolerance "
              "%.0e * (1 + mag)\n",
              minRatio, lowSteps, kRtol);

  // ---- the pieces alone
  {
    DevState tables(h.c, 1, std::vector<int>{1}, 1, true);
    t.check("pieces alone: lab 22.2's embed copies rows of E exactly (tokens 0, V-1, 7, 7)", [&] { return pieceEmbed(h); });
    t.check("pieces alone: lab X3.1's RMSNorm, 3 rows", [&] { return pieceRmsNorm(h); });
    for (Gemm g : kGemms) {
      if (!gemmPresent(g)) {
        std::printf("  (no %s: %s was not found when the lab was configured)\n", gemmName(g), gemmSource(g));
        continue;
      }
      t.check(std::string("pieces alone: ") + gemmSource(g) + ", batch 3, q's projection",
              [&] { return pieceLinear(h, g, 3); });
    }
    t.check("pieces alone: lab X3.1's RoPE, consecutive pairs (mode 0), positions 3 and seq_len - 1",
            [&] { return pieceRope(tables, h, LLAMA_ROPE_CONSECUTIVE); });
    t.check("pieces alone: lab X3.1's RoPE, half-split pairs (mode 1)", [&] { return pieceRope(tables, h, LLAMA_ROPE_HALF_SPLIT); });
    t.check("pieces alone: lab 20.4's KV append at rows 0 and seq_len - 1", [&] { return pieceKvAppend(h); });
    for (Attn at : {DECODE, GROUPED}) {
      if (!attnPresent(at)) {
        std::printf("  (no %s: not found when the lab was configured)\n", attnName(at));
        continue;
      }
      t.check(std::string("pieces alone: ") + attnName(at) + ", 3 and seq_len cached rows, the rest NaN",
              [&] { return pieceAttention(h, at); });
    }
    t.check("pieces alone: lab 2.3's add_residual", [&] { return pieceResidual(h); });
    t.check("pieces alone: lab X3.1's SwiGLU", [&] { return pieceSwiGLU(h); });
    t.check("pieces alone: lab 10.3's argmax, lowest index on a tie", [&] { return pieceArgmax(h); });
  }
  t.check("your advance alone: positions 0, 2 and 5 of a 4-token prompt", [&] { return checkAdvance(h.c); });

  // ---- your step
  // The linear layer: lab 22.1's gather on a GPU (the batch-1 kernel); on the emulators M3,
  // whose 128-wide tiles launch the fewest threads and so cost the emulator least (about a
  // quarter of the gather's time here), as lab 22.2 does. M2 if neither is present.
  const Gemm g0 = emu ? (gemmPresent(M3) ? M3 : M2) : (gemmPresent(GEMV) ? GEMV : M2);
  const Attn a0 = attnPresent(GROUPED) ? GROUPED : DECODE;
  const std::string with = std::string(" (") + gemmName(g0) + ", " + (a0 == GROUPED ? "grouped" : "decode") + " attention)";
  std::map<int, std::vector<int>> greedy;   // batch -> its record
  for (int B : batches) {
    std::vector<const RefTrace*> rs;
    for (int b = 0; b < B; ++b) rs.push_back(&refs[b]);
    Runner run(dm, g0, a0, LLAMA_ROPE_CONSECUTIVE);
    const std::string tf = "teacher-forced, batch " + std::to_string(B) + with;
    t.check(tf + ": every logit of every step within the tolerance", [&] { return teacherForced(run, dm, h, rs, steps, tf, true); });
    const std::string fr = "free-running greedy, batch " + std::to_string(B) + with;
    t.check(fr + ": " + std::to_string(steps) + " steps on the device, tokens equal the reference's", [&] {
      std::vector<int> rec;
      std::string e = generateRecord(run, h.c, rs, P, steps, &rec);
      if (!e.empty()) return e;
      int conclusive = 0;
      e = compareRecord(rec, rs, P, steps, fr, &conclusive);
      if (!e.empty()) return e;
      greedy[B] = rec;
      if (firstTime(fr + " tokens")) std::printf("    %s: sequence 0 generated %s\n", fr.c_str(), tokenLine(rec, B, 0, P - 1, steps).c_str());
      return weightsUnchanged(dm, h);
    });
  }
  t.check("batch invariance: each sequence of batch " + std::to_string(Bmax) + " gets the tokens it gets alone", [&] {
    if (!greedy.count(Bmax)) return std::string("not checked: the free-running check of that batch did not pass");
    Runner run(dm, g0, a0, LLAMA_ROPE_CONSECUTIVE);
    for (int b = 0; b < Bmax; ++b) {
      std::vector<int> alone;
      if (b == 0 && greedy.count(1)) {
        alone = greedy[1];                      // sequence 0 alone is the batch-1 check above
      } else {
        std::string e = generateRecord(run, h.c, {&refs[b]}, P, steps, &alone);
        if (!e.empty()) return "sequence " + std::to_string(b) + " alone: " + e;
      }
      for (int s = 0; s < steps; ++s)
        if (alone[s] != greedy[Bmax][(size_t)s * Bmax + b])
          return fmt("sequence %.0f, step %.0f: token %.0f in the batch, %.0f alone", b, s, greedy[Bmax][(size_t)s * Bmax + b],
                     alone[s]);
    }
    return std::string();
  });

  // Every other GEMM and attention kernel, teacher-forced at batch 1 (GPU only: the emulator
  // run would take minutes per kernel, and the pieces alone already ran each one).
  if (!emu) {
    for (Gemm g : kGemms)
      for (Attn at : {DECODE, GROUPED}) {
        if (!gemmPresent(g) || !attnPresent(at) || (g == g0 && at == a0)) continue;
        const std::string lab = std::string("teacher-forced, batch 1 (") + gemmName(g) + ", " +
                                (at == GROUPED ? "grouped" : "decode") + " attention)";
        t.check(lab + ": every logit within the tolerance", [&] {
          Runner run(dm, g, at, LLAMA_ROPE_CONSECUTIVE);
          return teacherForced(run, dm, h, {&refs[0]}, steps, lab, false);
        });
      }
  }

  // ---- RoPE pairing (B6): the permuted model under consecutive pairs computes what the
  // original computes under half-split pairs.
  {
    const RefTrace half = refRun(h, LLAMA_ROPE_HALF_SPLIT, refs[0].prompt, steps);
    if (firstTime("halfmargin")) {
      double r = 1e300;
      for (const RefStep& st : half.steps) r = std::min(r, st.gap / st.need);
      std::printf("  mode-1 reference (half-split RoPE): smallest top-two margin over twice the tolerance %.1f\n", r);
    }
    t.check("RoPE pairing, CPU reference: permuted model in mode 0 = original in mode 1 (to 1e-9)", [&] {
      const RefTrace perm = refRun(hp, LLAMA_ROPE_CONSECUTIVE, refs[0].prompt, steps);
      for (int s = 0; s < steps; ++s)
        for (int v = 0; v < h.c.vocab_size; ++v)
          if (!(std::fabs(perm.steps[s].logits[v] - half.steps[s].logits[v]) <= 1e-9 * (1 + half.steps[s].mag[v])))
            return fmt("step %.0f, token %.0f: %.12g vs %.12g", s, v, perm.steps[s].logits[v], half.steps[s].logits[v]);
      return std::string();
    });
    DevModel dmp(hp);
    if (!dmp.ok) {
      t.check("allocate the permuted model", [] { return std::string("allocation failed"); });
    } else {
      std::vector<int> recA, recB;
      for (int which = 0; which < 2; ++which) {
        DevModel& model = which ? dmp : dm;
        const HostModel& host = which ? hp : h;
        const int mode = which ? LLAMA_ROPE_CONSECUTIVE : LLAMA_ROPE_HALF_SPLIT;
        const std::string lab = which ? "RoPE pairing: the --permute-qk model in mode 0" : "RoPE pairing: the original model in mode 1";
        t.check(lab + ", teacher-forced: logits within the tolerance of the mode-1 reference", [&] {
          Runner run(model, g0, a0, mode);
          return teacherForced(run, model, host, {&half}, steps, lab + " (teacher-forced)", false);
        });
        if (emu) continue;   // the teacher-forced check above compared every step's token already
        t.check(lab + ", free-running: tokens equal the mode-1 reference's", [&] {
          Runner run(model, g0, a0, mode);
          std::vector<int>& rec = which ? recB : recA;
          std::string e = generateRecord(run, host.c, {&half}, P, steps, &rec);
          if (!e.empty()) return e;
          int conclusive = 0;
          return compareRecord(rec, {&half}, P, steps, lab + " (free-running)", &conclusive);
        });
      }
      if (!emu && recA.size() == (size_t)steps && recB.size() == (size_t)steps && firstTime("ropepair"))
        std::printf("    mode 1 tokens: %s\n    mode 0, permuted: %s\n", tokenLine(recA, 1, 0, P - 1, steps).c_str(),
                    tokenLine(recB, 1, 0, P - 1, steps).c_str());
    }
  }

  // ---- the sampler (M5)
#if defined(PMPP_HAVE_M5)
  if (!emu) {
    std::vector<const RefTrace*> rs;
    for (int b = 0; b < Bmax; ++b) rs.push_back(&refs[b]);
    t.check("sampler (M5) at temperature 0: tokens equal the greedy step's", [&] {
      if (!greedy.count(Bmax)) return std::string("not checked: the greedy free-running check did not pass");
      SamplerSettings set;
      set.temperature = 0.0f;
      SamplerHolder hold(Bmax, h.c.vocab_size, set);
      if (!hold.problem.empty()) return hold.problem;
      Runner run(dm, g0, a0, LLAMA_ROPE_CONSECUTIVE);
      run.sampler = &hold.ctx;
      std::vector<int> rec;
      std::string e = generateRecord(run, h.c, rs, P, steps, &rec);
      if (!e.empty()) return e;
      for (size_t i = 0; i < rec.size(); ++i)
        if (rec[i] != greedy[Bmax][i]) return fmt("step %.0f, sequence %.0f: sampler token differs from the argmax", (double)(i / Bmax), (double)(i % Bmax));
      return std::string();
    });
    t.check("sampler (M5) at temperature 0.8, top-k 40, top-p 0.95, min-p 0.05: two runs, identical tokens", [&] {
      if (!greedy.count(Bmax)) return std::string("not checked: the greedy free-running check did not pass");
      SamplerSettings set;
      std::vector<int> rec[2];
      for (int r = 0; r < 2; ++r) {
        SamplerHolder hold(Bmax, h.c.vocab_size, set);
        if (!hold.problem.empty()) return hold.problem;
        Runner run(dm, g0, a0, LLAMA_ROPE_CONSECUTIVE);
        run.sampler = &hold.ctx;
        std::string e = generateRecord(run, h.c, rs, P, steps, &rec[r]);
        if (!e.empty()) return e;
      }
      if (rec[0] != rec[1]) return std::string("the two runs differ: the sampler is not deterministic per (seed, stream, step)");
      if (firstTime("sampled")) {
        int differ = 0;
        for (size_t i = (size_t)(P - 1) * Bmax; i < rec[0].size(); ++i) differ += rec[0][i] != greedy[Bmax][i];
        std::printf("    sampled, sequence 0: %s\n    %d of %d generated tokens differ from greedy (not a test)\n",
                    tokenLine(rec[0], Bmax, 0, P - 1, steps).c_str(), differ, G * Bmax);
      }
      return std::string();
    });
  } else {
    std::printf("  (the sampler checks run on GPUs only: lab X1.3 tests the sampler on the emulators)\n");
  }
#else
  std::printf("  (no sampler checks: lab X1.3's solution was not found when the lab was configured)\n");
#endif
  return t.finish();
}

// ---------------------------------------------------------------- benchmark
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

// Lab 5.1's measured bandwidth on this backend, from reports/runs/<backend>.json, or 0.
static double lab51Bandwidth() {
  std::FILE* f = std::fopen((std::string(PMPP_X3_REPORTS) + "/" + pmpp::backend() + ".json").c_str(), "rb");
  if (!f) return 0;
  std::string s;
  char buf[65536];
  size_t n;
  while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
  std::fclose(f);
  double best = 0;
  for (size_t at = s.find("\"bandwidth\""); at != std::string::npos; at = s.find("\"bandwidth\"", at + 1)) {
    const size_t open = s.rfind('{', at), close = s.find('}', at);
    if (open == std::string::npos || close == std::string::npos) continue;
    const std::string row = s.substr(open, close - open);
    if (row.find("ch05/lab1_roofline") == std::string::npos) continue;
    const size_t k = row.find("\"gb_per_s\"");
    if (k == std::string::npos) continue;
    const size_t colon = row.find(':', k);
    best = std::max(best, std::atof(row.c_str() + colon + 1));
  }
  return best;
}

enum Piece { P_EMBED, P_RMS, P_Q, P_K, P_V, P_ROPE_Q, P_ROPE_K, P_APPEND, P_ATT, P_O, P_RES, P_W1, P_W3, P_SWIGLU, P_W2,
             P_HEAD, P_ARGMAX, P_ADVANCE, P_COUNT };
static const char* kPieceNames[P_COUNT] = {"embed", "rmsnorm", "q", "k", "v", "rope q", "rope k", "kv append", "attention",
                                          "wo", "residual", "w1", "w3", "swiglu", "w2", "lm head", "argmax", "advance"};
// How often a step runs each piece, and the group the split reports it in.
static int pieceCount(Piece p, int L) {
  switch (p) {
    case P_EMBED: case P_HEAD: case P_ARGMAX: case P_ADVANCE: return 1;
    case P_RMS: return 2 * L + 1;
    case P_RES: return 2 * L;
    default: return L;
  }
}
enum Group { G_MATMUL, G_ATTN, G_ELEM, G_ENDS, G_COUNT };
static const char* kGroupNames[G_COUNT] = {"matmuls", "attention", "norms+rope+swiglu+residual",
                                          "embed+lm head+argmax+advance"};
static Group groupOf(Piece p) {
  switch (p) {
    case P_Q: case P_K: case P_V: case P_O: case P_W1: case P_W3: case P_W2: return G_MATMUL;
    case P_APPEND: case P_ATT: return G_ATTN;
    case P_RMS: case P_ROPE_Q: case P_ROPE_K: case P_RES: case P_SWIGLU: return G_ELEM;
    default: return G_ENDS;
  }
}
static pmpp::Timing runPiece(Piece p, Gemm g, Attn a, DevModel& dm, DevState& s, std::string* err) {
  const LlamaConfig& c = dm.c;
  const int B = s.batch, d = c.dim, h = c.hidden_dim, dq = dqOf(c), dkv = dkvOf(c);
  auto& w = dm.L[0];
  switch (p) {
    case P_EMBED: return opEmbed(dm.E, s.tokens, s.x, B, d, err);
    case P_RMS: return opRmsNorm(s.x, w[W_RMS_ATT], s.xb, B, d, 1e-5f, err);
    case P_Q: return opLinear(g, s.xb, w[W_Q], s.q, B, d, dq, err);
    case P_K: return opLinear(g, s.xb, w[W_K], s.k, B, d, dkv, err);
    case P_V: return opLinear(g, s.xb, w[W_V], s.v, B, d, dkv, err);
    case P_ROPE_Q: return opRope(s.q, s.cosT, s.sinT, s.pos, B, c.n_heads, LLAMA_HEAD_DIM, 0, err);
    case P_ROPE_K: return opRope(s.k, s.cosT, s.sinT, s.pos, B, c.n_kv_heads, LLAMA_HEAD_DIM, 0, err);
    case P_APPEND: return opKvAppend(s.k, s.v, s.Kc[0], s.Vc[0], s.pos, B, c.seq_len, c.n_kv_heads, err);
    case P_ATT:
      return opAttention(a, s.q, s.Kc[0], s.Vc[0], s.len, s.att, B, c.n_heads, c.n_kv_heads, c.seq_len,
                         1.0f / std::sqrt((float)LLAMA_HEAD_DIM), err);
    case P_O: return opLinear(g, s.att, w[W_O], s.xb2, B, dq, d, err);
    case P_RES: return opResidual(s.x, s.xb2, B * d, err);
    case P_W1: return opLinear(g, s.xb, w[W_1], s.hb, B, d, h, err);
    case P_W3: return opLinear(g, s.xb, w[W_3], s.hb2, B, d, h, err);
    case P_SWIGLU: return opSwiGLU(s.hb, s.hb2, s.hb, B * h, err);
    case P_W2: return opLinear(g, s.hb, w[W_2], s.xb2, B, h, d, err);
    case P_HEAD: return opLinear(g, s.xb, dm.E, s.logits, B, d, c.vocab_size, err);
    case P_ARGMAX: return opArgmax(s.logits, s.tokens, B, c.vocab_size, err);
    default: return opAdvance(s, err);
  }
}

static int bench(const pmpp::Args& a) {
  // The shape (ours, about a 0.4B-parameter Llama): halve d, the heads and the hidden width
  // until the weights take at most a quarter of the memory the device reports.
  LlamaConfig c{2048, 5632, 8, 16, 4, 32000, 1024};
  const double mem = deviceMemory();
  while (c.dim > 256 && mem > 0 && weightBytes(c) > mem / 4) {
    c.dim /= 2;
    c.n_heads = c.dim / LLAMA_HEAD_DIM;
    c.n_kv_heads = std::max(1, c.n_heads / 4);
    c.hidden_dim /= 2;
  }
  const std::string bin = std::string(PMPP_X3_MODEL_DIR) + "/bench.bin";
  char cmd[2048];
  std::snprintf(cmd, sizeof cmd,
                "\"%s\" \"%s\" --dim %d --hidden %d --layers %d --heads %d --kv-heads %d --vocab %d --seq-len %d "
                "--seed 7 --fill tile --out \"%s\"",
                PMPP_X3_PYTHON, PMPP_X3_MAKE_MODEL, c.dim, c.hidden_dim, c.n_layers, c.n_heads, c.n_kv_heads,
                c.vocab_size, c.seq_len, bin.c_str());
  std::printf("writing the benchmark model with make_model.py (tiled fill: its numbers mean nothing)\n");
  std::fflush(stdout);
  if (std::system(cmd) != 0) {
    std::printf("make_model.py failed\n");
    return 1;
  }
  std::unique_ptr<DevModel> dm;
  {
    HostModel h;
    std::string e = loadModel(bin, h);
    std::remove(bin.c_str());   // 4 bytes per weight on disk; written again at the next benchmark
    if (!e.empty()) {
      std::printf("%s\n", e.c_str());
      return 1;
    }
    dm.reset(new DevModel(h));
  }
  if (!dm->ok) {
    std::printf("could not allocate the weights (%s)\n", shape(c).c_str());
    return 1;
  }
  const double wb = weightBytes(c);
  std::printf("model: %s, float32, E tied to the LM head\n       weights %.1f MB read per step; timing: %s\n",
              shape(c).c_str(), wb / 1e6, stepUnit());
  pmpp::bench_row(a, "model").n("d", c.dim).n("hidden", c.hidden_dim).n("layers", c.n_layers).n("heads", c.n_heads)
      .n("kv_heads", c.n_kv_heads).n("vocab", c.vocab_size).n("seq_len", c.seq_len).n("weight_bytes", wb).emit();

  const Attn att = attnPresent(GROUPED) ? GROUPED : DECODE;
  const int positions[2] = {16, 512};
  const int batchesB[2] = {1, 4};
  const double bw51 = lab51Bandwidth();
  std::map<std::string, double> tps;   // "gemm/batch/pos" -> tokens per second

  // ---- 'step' rows
  for (Gemm g : kGemms) {
    if (!gemmPresent(g)) {
      std::printf("\nno %s rows: %s was not found when the lab was configured\n", gemmName(g), gemmSource(g));
      continue;
    }
    Runner run(*dm, g, att, LLAMA_ROPE_CONSECUTIVE);
    if (!run.problem.empty()) {
      std::printf("\nno %s rows: %s\n", gemmName(g), run.problem.c_str());
      continue;
    }
    std::printf("\n%s (%s), %s: one decode step\n", gemmName(g), gemmSource(g), attnName(att));
    std::printf("  batch  position    step ms   tokens/s   weight MB/token   weight GB/s   KV MB read\n");
    for (int pi = 0; pi < 2; ++pi)
      for (int bi = 0; bi < 2; ++bi) {
        const int B = batchesB[bi], p = positions[pi];
        if (g == GEMV && B > kGemvMaxBatch) continue;
        DevState s(c, B, std::vector<int>(B, 1), 1, false);
        if (!s.ok) {
          pmpp::take_error();
          std::printf("  %5d  %8d    allocation failed\n", B, p);
          continue;
        }
        // The cache rows before p hold finite values (a prompt-filled cache; their values do
        // not change the time). Each timed step appends row p and is then put back at p.
        std::vector<float> fill = rnd(s.cacheWords(), 5, 0.05f);
        for (int b = 0; b < B; ++b)
          for (size_t i = ((size_t)b * c.seq_len + p) * dkvOf(c); i < (size_t)(b + 1) * c.seq_len * dkvOf(c); ++i) fill[i] = 0;
        for (int l = 0; l < c.n_layers; ++l) {
          uploadGuarded(s.Kc[l], fill);
          uploadGuarded(s.Vc[l], fill);
        }
        bool ok = true;
        pmpp::Stat st = measure(a, [&] {
          s.setPositions(p, std::vector<int>(B, 1 + p % 97));
          return run.step(s, &ok);
        });
        std::string err = pmpp::take_error();
        if (!err.empty() || !ok || st.min_s <= 0) {
          std::printf("  %5d  %8d    the step failed (%s)\n", B, p, err.empty() ? "a piece refused to dispatch" : err.c_str());
          continue;
        }
        const double sec = st.min_s, kv = 2.0 * 4 * c.n_layers * B * (p + 1.0) * dkvOf(c);
        tps[std::string(gemmName(g)) + "/" + std::to_string(B) + "/" + std::to_string(p)] = B / sec;
        std::printf("  %5d  %8d  %9.3f  %9.1f   %15.2f   %11.1f   %10.2f\n", B, p, sec * 1e3, B / sec, wb / B / 1e6,
                    wb / sec / 1e9, kv / 1e6);
        pmpp::bench_row(a, "step").s("gemm", gemmName(g)).s("attention", att == GROUPED ? "grouped" : "decode")
            .n("batch", B).n("position", p).n("seconds", sec).n("median_s", st.median_s).n("tokens_per_s", B / sec)
            .n("weight_bytes_per_token", wb / B).n("weight_gb_per_s", wb / sec / 1e9).n("kv_bytes", kv).emit();
      }
  }

  // ---- 'piece' rows: each piece alone, times how often a step runs it, grouped
  const Gemm g0 = gemmPresent(GEMV) ? GEMV : M2;
  {
    Runner probe(*dm, g0, att, LLAMA_ROPE_CONSECUTIVE);
    if (probe.problem.empty()) {
      std::printf("\n%s: each piece alone, times how often one step runs it (%d layers), ms\n", gemmName(g0), c.n_layers);
      std::printf("  batch  position");
      for (int gi = 0; gi < G_COUNT; ++gi) std::printf("  %s", kGroupNames[gi]);
      std::printf("   sum of pieces   attention share\n");
      for (int pi = 0; pi < 2; ++pi)
        for (int bi = 0; bi < 2; ++bi) {
          const int B = batchesB[bi], p = positions[pi];
          DevState s(c, B, std::vector<int>(B, 1), 1, false);
          if (!s.ok) {
            pmpp::take_error();
            continue;
          }
          std::vector<float> fill = rnd(s.cacheWords(), 5, 0.05f);
          uploadGuarded(s.Kc[0], fill);
          uploadGuarded(s.Vc[0], fill);
          double groups[G_COUNT] = {0, 0, 0, 0};
          std::string err;
          for (int q = 0; q < P_COUNT && err.empty(); ++q) {
            pmpp::Stat st = measure(a, [&] {
              s.setPositions(p, std::vector<int>(B, 1 + p % 97));
              return runPiece((Piece)q, g0, att, *dm, s, &err);
            });
            if (err.empty()) err = pmpp::take_error();
            groups[groupOf((Piece)q)] += st.min_s * pieceCount((Piece)q, c.n_layers);
          }
          if (!err.empty()) {
            std::printf("  %5d  %8d  failed (%s)\n", B, p, err.c_str());
            continue;
          }
          const double sum = groups[0] + groups[1] + groups[2] + groups[3];
          std::printf("  %5d  %8d", B, p);
          for (int gi = 0; gi < G_COUNT; ++gi) std::printf("  %*.3f", (int)std::strlen(kGroupNames[gi]), groups[gi] * 1e3);
          std::printf("   %13.3f   %15.1f%%\n", sum * 1e3, 100.0 * groups[G_ATTN] / sum);
          for (int gi = 0; gi < G_COUNT; ++gi)
            pmpp::bench_row(a, "piece").s("gemm", gemmName(g0)).n("batch", B).n("position", p).s("group", kGroupNames[gi])
                .n("seconds", groups[gi]).n("share", groups[gi] / sum).emit();
        }
      std::printf("  (each piece is one launch or command buffer of its own here, so the sum can exceed a step)\n");
    }
  }

  // ---- the ceiling: lab 5.1's bandwidth over the weight bytes per token
  const auto it = tps.find(std::string(gemmName(g0)) + "/1/16");
  if (bw51 > 0) {
    const double ceiling = bw51 * 1e9 / wb;
    std::printf("\nlab 5.1 measured %.1f GB/s here: at batch 1 the weights alone allow %.1f tokens/s", bw51, ceiling);
    if (it != tps.end()) std::printf("; %s reached %.1f (%.0f%% of it)", gemmName(g0), it->second, 100.0 * it->second / ceiling);
    std::printf("\n");
    pmpp::bench_row(a, "ceiling").n("lab51_gb_per_s", bw51).n("weight_bytes_per_token", wb)
        .n("ceiling_tokens_per_s", ceiling).n("tokens_per_s", it != tps.end() ? it->second : NAN)
        .n("fraction", it != tps.end() ? it->second / ceiling : NAN).emit();
  } else {
    std::printf("\nno lab 5.1 bandwidth in reports/runs/%s.json: run `python3 labs/pmpp.py run 5.1 --bench` for the ceiling\n",
                pmpp::backend());
  }
  std::printf("\nThese are measurements of this machine, not expectations.\n");
  return 0;
}

int main(int argc, char** argv) { return pmpp::lab_main(argc, argv, tests, bench); }
